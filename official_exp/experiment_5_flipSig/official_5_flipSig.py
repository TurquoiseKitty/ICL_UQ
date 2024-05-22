from train import main, config_save_special
from quinine import QuinineArgumentParser, Quinfig
from multigpu_utils import init_distributed_mode
from schema import schema
import uuid
import os
import yaml
import argparse
from tasks import LinearRegression, get_task_sampler
import numpy as np
import torch
import copy
import torch.distributed as dist
import time
from samplers import get_data_sampler

def list_of_strings(arg):
    return arg.split(',')


def mass_parser():

    parser = argparse.ArgumentParser(
        description="a light parser that leverages quinine parse"
    )
    parser.add_argument(
        "--quinine_config_path",
        type=str,
    )
    parser.add_argument('--local_rank', default=0, type=int)
    # parser.add_argument('--local-rank', default=-1, type=int) # just a holder
    # parser.add_argument('--mask_config', type=str)
    # parser.add_argument('--mask_scheme', type=str)
    parser.add_argument('--pool_size', type=int, default=0)

    return parser




if __name__ == "__main__":

    raw_args = mass_parser().parse_args()


    pool_size = raw_args.pool_size


    template_args = Quinfig(config_path=raw_args.quinine_config_path, schema=schema)

    template_args.training.task_kwargs["exempt_mode"] = True

    
    
    # all_poolSizes = [
    #     64, 256, 1024, 4096, 16192
    # ]

    init_distributed_mode(template_args.multigpu)
        
    pool_of_w =  LinearRegression.generate_pool_dict(
        n_dims =  template_args.model.n_dims,
        num_tasks =  pool_size)["w"]
    
    # first part is all positive, and low sig
    # second part is all negative, large sig

    pool_of_w[:int(pool_size/2)] = torch.abs(pool_of_w[:int(pool_size/2)])
    pool_of_w[int(pool_size/2):] = -torch.abs(pool_of_w[int(pool_size/2):])


    pool_of_sigma = torch.Tensor(np.random.uniform(low=0.1, high=0.3, size=pool_size))
    pool_of_sigma[int(pool_size/4):int(pool_size/2)] = torch.Tensor(np.random.uniform(low=0.5, high=0.7, size=pool_size))[int(pool_size/4):int(pool_size/2)]
    pool_of_sigma[int(pool_size/2):int(3*pool_size/4)] = torch.Tensor(np.random.uniform(low=0.3, high=0.5, size=pool_size))[int(pool_size/2):int(3*pool_size/4)]
    pool_of_sigma[int(3*pool_size/4):] = torch.Tensor(np.random.uniform(low=0.7, high=0.9, size=pool_size))[int(3*pool_size/4):]

    pool_of_exempt = np.zeros(pool_size)

    args = copy.deepcopy(template_args)

    args.training.task_kwargs["pool_dict"] = {
        "w": pool_of_w,
        "sigma": pool_of_sigma,
        "exempt": pool_of_exempt
    }


    # ------------------don't forget------------------------------
    exp_name = args.wandb.name + "_poolSize_" + str(pool_size)
    args.wandb.name = exp_name
    # ------------------don't forget------------------------------

    # add mask
    max_sampe_len = args.training.curriculum.points.end + 1

    mask = np.ones(max_sampe_len)
    
    args.training.task_kwargs["mask"] = mask

    run_id = args.training.resume_id
    if run_id is None:
        # run_id = str(uuid.uuid4())
        run_id = exp_name

    out_dir = os.path.join(args.out_dir, run_id)

    args.out_dir = out_dir

    assert args.model.family in ["gpt2", "lstm"]
    # print(f"Running with: {args}")

    if not args.test_run:
        if ( not args.multigpu.distributed ) or (args.multigpu.local_rank == 0):
            
            if not os.path.exists(args.out_dir):
                os.makedirs(args.out_dir)
            
            raw_dic = copy.deepcopy(args.__dict__)
            config_save_special(out_dir, raw_dic)

    
    def panel_plot_func(eval_model):


        w_group = ["pos", "neg"]
        sig_group = ["0103", "0305", "0507", "0709"]
        step_group = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        output = {}

        N_sample = 200

        data_sampler = get_data_sampler(args.training.data, n_dims=args.model.n_dims)

        eval_xs = data_sampler.sample_xs(
            args.training.curriculum.points.end,
            N_sample,
            args.training.curriculum.points.end,
        ).cuda()


        # generate evaluate data
        for w_sign in w_group:

            pool_of_w = torch.randn(N_sample, args.model.n_dims, 1).cuda()

            if w_sign == "pos":

                pool_of_w = torch.abs(pool_of_w)

            elif w_sign == "neg":
            
                pool_of_w = -torch.abs(pool_of_w)    

            for sig_range in sig_group:

                if sig_range == "0103":
                    low_sig = 0.1
                    high_sig = 0.3
                elif sig_range == "0305":
                    low_sig = 0.3
                    high_sig = 0.5
                elif sig_range == "0507":
                    low_sig = 0.5
                    high_sig = 0.7
                elif sig_range == "0709":
                    low_sig = 0.7
                    high_sig = 0.9

                pool_of_sigma = torch.Tensor(np.random.uniform(low=low_sig, high=high_sig, size=N_sample)).cuda()

                base_config = copy.deepcopy(args)

                base_config["training"]["task_kwargs"]["pool_dict"]["w"] = pool_of_w
                base_config["training"]["task_kwargs"]["pool_dict"]["sigma"] = pool_of_sigma
                base_config["training"]["task_kwargs"]["pool_dict"]["exempt"] = np.zeros(N_sample)

                eval_task_sampler = get_task_sampler(
                    base_config.training.task,
                    args.model.n_dims,
                    N_sample,
                    **base_config.training.task_kwargs
                )

                eval_task = eval_task_sampler()

                eval_mus, eval_ys, eval_sigma, eval_exemps = eval_task.eval_with_mu_sigma_exempt(eval_xs)

                with torch.no_grad():
                    eval_pred = eval_model(eval_xs, eval_ys)

                # start the recording
                    
                eval_predmu = eval_pred[:, :, 0]
                eval_predsigma = eval_pred[:, :, 1]

                def softplus(arr):
                    return np.log(1+np.exp(arr))
                
                eval_mu_diff = np.abs(eval_predmu.cpu().numpy()-eval_mus.cpu().numpy()).mean(axis=0)


                real_eval_pred_sigma = softplus(eval_predsigma.cpu().numpy())

                eval_full_sigma  = np.tile(eval_sigma.cpu().numpy().reshape(-1,1), (1, eval_predsigma.shape[1]))

                eval_sigma_diff = np.abs(real_eval_pred_sigma-eval_full_sigma).mean(axis=0)

                for obs_step in step_group:

                    mu_loss_name = "AbsDev_mu_"+ w_sign + "_sig_" + sig_range + "_at_" + str(obs_step)
                    sig_loss_name = "AbsDev_sigma_"+ w_sign + "_sig_" + sig_range + "_at_" + str(obs_step)
                    percent0103 = "sigpercent_0103_"+ w_sign + "_sig_" + sig_range + "_at_" + str(obs_step)
                    percent0305 = "sigpercent_0305_"+ w_sign + "_sig_" + sig_range + "_at_" + str(obs_step)
                    percent0507 = "sigpercent_0507_"+ w_sign + "_sig_" + sig_range + "_at_" + str(obs_step)
                    percent0709 = "sigpercent_0709_"+ w_sign + "_sig_" + sig_range + "_at_" + str(obs_step)

                    percent_in_0103 =  sum(0.1 <= sigs <= 0.3 for sigs in real_eval_pred_sigma[:, obs_step]) / len(real_eval_pred_sigma[:, obs_step])
                    percent_in_0305 =  sum(0.3 <= sigs <= 0.5 for sigs in real_eval_pred_sigma[:, obs_step]) / len(real_eval_pred_sigma[:, obs_step])

                    percent_in_0507 =  sum(0.5 <= sigs <= 0.7 for sigs in real_eval_pred_sigma[:, obs_step]) / len(real_eval_pred_sigma[:, obs_step])

                    percent_in_0709 =  sum(0.7 <= sigs <= 0.9 for sigs in real_eval_pred_sigma[:, obs_step]) / len(real_eval_pred_sigma[:, obs_step])

                    output[mu_loss_name] = eval_mu_diff[obs_step]
                    output[sig_loss_name] = eval_sigma_diff[obs_step]
                    output[percent0103] = percent_in_0103
                    output[percent0305] = percent_in_0305
                    output[percent0507] = percent_in_0507
                    output[percent0709] = percent_in_0709


        return output


    main(args, use_wandb= True, panel_plot=panel_plot_func) 




        

