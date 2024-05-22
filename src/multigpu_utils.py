import builtins
import datetime
import os
import time
import torch
import torch.distributed as dist
import numpy as np


def init_distributed_mode(args):
    if args.dist_on_itp:

        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:

        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:

        args.rank = int(os.environ['SLURM_PROCID'])
        args.local_rank = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_print(is_master=True)  # hack
        args.distributed = False
        return
    
    if args.world_size == 1:
        print('Not actually using distributed mode')
        setup_print(is_master=True)  # hack
        args.distributed = False
        return
    
    args.distributed = True
    
    torch.cuda.set_device(args.local_rank)

    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.local_rank), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    
    torch.distributed.barrier()
    setup_print(args.rank == 0)



def agg_ListOfStr_print(list_of_str, local_rank, world_size):

    organizer = {}

    for idx in range(world_size):

        organizer[str(idx)] = []


    for stri in list_of_str:

        hold_list = [None for _ in range(world_size)]

        dist.all_gather_object(hold_list, stri)

        for i in range(world_size):

            organizer[str(i)].append(hold_list[i])

    # print
    if local_rank == 0:

        for rank in range(world_size):

            print("************************************")
            print("rank: ", rank)
            for stri in organizer[str(rank)]:

                print("    "+stri)
            print("************************************")

        





def setup_print(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()
