import os
from random import randint
import uuid
import copy
import numpy as np

from quinine import QuinineArgumentParser, Quinfig
from tqdm import tqdm
import torch
import yaml
import argparse

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model
from multigpu_utils import init_distributed_mode

import wandb
from munch import Munch
from utils.misc import seed_all
from eval import get_model_from_run

torch.backends.cudnn.benchmark = True


def train_step(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()
    output = model(xs, ys)
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args, use_wandb = True, panel_plot = None, explicit_seed = None):

    if explicit_seed is not None:
        seed_all(explicit_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = args.model.n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)

    # ----------------------------------------construct task sampler---------------
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
    )
    # ----------------------------------------construct task sampler---------------

    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples

    current_seed = None

    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        if explicit_seed is not None:
            current_seed = explicit_seed + i * args.multigpu.world_size

        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            explicit_seed = current_seed,
            **data_sampler_args,
        )
        
        task = task_sampler(explicit_seed = current_seed, **task_sampler_args)

        if args.training.task_kwargs["exempt_mode"] and args.model.uncertainty_model:


            _, ys, exempt_indicator = task.eval_with_mu_exempt(xs, explicit_seed = current_seed)

            loss_func = task.get_training_metric(exempt_indicator = exempt_indicator, **args.training.task_kwargs)

        else:
            ys = task.evaluate(xs)

            loss_func = task.get_training_metric(**args.training.task_kwargs)

        loss, output = train_step(model, xs.cuda(), ys.cuda(), optimizer, loss_func)

        # point_wise_tags = list(range(curriculum.n_points))
        # point_wise_loss_func = task.get_metric()
        # point_wise_loss = point_wise_loss_func(output, ys.to(output.device)).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        curriculum.update()

        if use_wandb and (( not args.multigpu.distributed ) or (args.multigpu.local_rank == 0)):

            if i % args.wandb.log_every_steps == 0 and not args.test_run:
                wandb.log(
                    {
                        "overall_loss": loss,
                        "n_points": curriculum.n_points,
                        "n_dims": curriculum.n_dims_truncated,
                    },
                    step=i,
                )

                if (curriculum.n_points >= curriculum.n_points_schedule.end) and (curriculum.n_dims_truncated >= curriculum.n_dims_schedule.end):

                    ## special logs for the transient experiment
                    if (args.wandb.name in ["TransientNature", "VT_10", "LargeWTransition"]) or ("PoolCtrlFlip_" in args.wandb.name) or ("PoolUsingLargeW_" in args.wandb.name) or ("maskInitPhase_" in args.wandb.name) or ("maskVarious_" in args.wandb.name):

                        w_group = ["pos", "neg"]
                        sig_group = ["0103", "0305", "0507", "0709"]
                        step_group = [10, 20, 40, 60]

                        
                        eval_model = model.eval()

                        N_sample = 1000

                        eval_xs = data_sampler.sample_xs(
                            curriculum.n_points,
                            N_sample,
                            curriculum.n_dims_truncated,
                            **data_sampler_args,
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
                                    n_dims,
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

                                    wandb.log(
                                        {
                                            mu_loss_name: eval_mu_diff[obs_step],
                                            sig_loss_name: eval_sigma_diff[obs_step],
                                            percent0103: percent_in_0103,
                                            percent0305: percent_in_0305,
                                            percent0507: percent_in_0507,
                                            percent0709: percent_in_0709,
                                        },
                                        step=i,
                                    )

                    elif panel_plot is not None:
                        log_dic = panel_plot(model.eval())
                        wandb.log(log_dic, step=i)


            pbar.set_description(f"loss {loss}")
            if i % args.training.save_every_steps == 0 and not args.test_run:
                training_state = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_step": i,
                }
                torch.save(training_state, state_path)

            if (
                args.training.keep_every_steps > 0
                and i % args.training.keep_every_steps == 0
                and not args.test_run
                and i > 0
            ):
                torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args, use_wandb = True, panel_plot=None, add_embed=False, random_embedding=None, explicit_seed = None,
         special_build_multiply = None,
         old_model_path = None,
         inverse_model_path = None):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    elif (use_wandb) and  (( not args.multigpu.distributed ) or (args.multigpu.local_rank == 0)):

        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    model = build_model(args.model, add_embed=add_embed, random_embedding=random_embedding, explicit_seed = explicit_seed)

    if special_build_multiply is not None:

        old_model, _ = get_model_from_run(old_model_path)
        inverse_model, _ = get_model_from_run(inverse_model_path)
        with torch.no_grad():
            # Copy the parameters of model1 and model2
            theta_1 = old_model.state_dict()
            theta_2 = inverse_model.state_dict()

            # Create the parameters for the new model as 2 * theta_1 - theta_2
            new_parameters = {}
            for name in theta_1:
                new_parameters[name] = special_build_multiply * theta_1[name] - theta_2[name]

            # Load the new parameters into the new model
            model.load_state_dict(new_parameters)



    if not args.multigpu.distributed:
        model.cuda()
    else:
        model.to(device = torch.device("cuda", args.multigpu.local_rank))
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.multigpu.local_rank], output_device=args.multigpu.local_rank,
            find_unused_parameters=True)
        
    model.train()

    train(model, args, use_wandb = use_wandb, panel_plot= panel_plot, explicit_seed=explicit_seed)

    # if not args.test_run:
    #     _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


    if use_wandb and (( not args.multigpu.distributed ) or (args.multigpu.local_rank == 0)):

        wandb.finish()


    




def light_parser():

    parser = argparse.ArgumentParser(
        description="a light parser that leverages quinine parse"
    )
    parser.add_argument(
        "--quinine_config_path",
        type=str,
    )
    parser.add_argument('--local_rank', default=0, type=int)
    # parser.add_argument('--local-rank', default=-1, type=int) # just a holder

    return parser

def config_save_special(out_dir, raw_dic):
              
    if "task_kwargs" in raw_dic["training"].keys() and "pool_dict" in raw_dic["training"]["task_kwargs"].keys():
        if "w" in raw_dic["training"]["task_kwargs"]["pool_dict"].keys():
            pool_w = raw_dic["training"]["task_kwargs"]["pool_dict"]["w"].numpy()
            raw_dic["training"]["task_kwargs"]["pool_dict"]["w"] = None
            np.save(os.path.join(out_dir, "config_pool_w.npy"), pool_w)
        if "sigma" in raw_dic["training"]["task_kwargs"]["pool_dict"].keys():
            pool_sigma = raw_dic["training"]["task_kwargs"]["pool_dict"]["sigma"].numpy()
            raw_dic["training"]["task_kwargs"]["pool_dict"]["sigma"] = None
            np.save(os.path.join(out_dir, "config_pool_sigma.npy"), pool_sigma)
        if "exempt" in raw_dic["training"]["task_kwargs"]["pool_dict"].keys():
            pool_exempt = raw_dic["training"]["task_kwargs"]["pool_dict"]["exempt"]
            raw_dic["training"]["task_kwargs"]["pool_dict"]["exempt"] = None
            np.save(os.path.join(out_dir, "config_pool_exempt.npy"), pool_exempt)

    if "task_kwargs" in raw_dic["training"].keys() and "mask" in raw_dic["training"]["task_kwargs"].keys():
        mask = raw_dic["training"]["task_kwargs"]["mask"]
        raw_dic["training"]["task_kwargs"]["mask"] = None
        np.save(os.path.join(out_dir, "config_mask.npy"), mask)

    with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
                
        yaml.dump(raw_dic, yaml_file, default_flow_style=False)
            

def config_load_special(load_dir, 
                        mask_flag = True, pool_w_flag = True, pool_sigma_flag = True, pool_exempt_flag = True):
    
    config_path = os.path.join(load_dir, "config.yaml")
    with open(config_path) as fp:  # we don't Quinfig it to avoid inherits
        conf = Munch.fromDict(yaml.safe_load(fp))

    if mask_flag:
        mask_path = os.path.join(load_dir, "config_mask.npy")
        mask = np.load(mask_path)
        conf["training"]["task_kwargs"]["mask"] = mask

    if pool_w_flag:
        pool_w_path = os.path.join(load_dir, "config_pool_w.npy")
        pool_w = np.load(pool_w_path)
        conf["training"]["task_kwargs"]["pool_dict"]["w"] = torch.Tensor(pool_w)

    if pool_sigma_flag:
        pool_sigma_path = os.path.join(load_dir, "config_pool_sigma.npy")
        pool_sigma = np.load(pool_sigma_path)
        conf["training"]["task_kwargs"]["pool_dict"]["sigma"] = torch.Tensor(pool_sigma)

    if pool_exempt_flag:
        pool_exempt_path = os.path.join(load_dir, "config_pool_exempt.npy")
        pool_exempt = np.load(pool_exempt_path)
        conf["training"]["task_kwargs"]["pool_dict"]["exempt"] = pool_exempt

    return conf     



def config_save_tri(out_dir, raw_dic):
              
    if "task_kwargs" in raw_dic["training"].keys() and "pool_dict" in raw_dic["training"]["task_kwargs"].keys():
        if "reg_w" in raw_dic["training"]["task_kwargs"]["pool_dict"].keys():
            reg_w = raw_dic["training"]["task_kwargs"]["pool_dict"]["reg_w"].numpy()
            raw_dic["training"]["task_kwargs"]["pool_dict"]["reg_w"] = None
            np.save(os.path.join(out_dir, "config_reg_w.npy"), reg_w)
        if "reg_sigma" in raw_dic["training"]["task_kwargs"]["pool_dict"].keys():
            reg_sigma = raw_dic["training"]["task_kwargs"]["pool_dict"]["reg_sigma"].numpy()
            raw_dic["training"]["task_kwargs"]["pool_dict"]["reg_sigma"] = None
            np.save(os.path.join(out_dir, "config_reg_sigma.npy"), reg_sigma)
        if "reg_exempt" in raw_dic["training"]["task_kwargs"]["pool_dict"].keys():
            reg_exempt = raw_dic["training"]["task_kwargs"]["pool_dict"]["reg_exempt"]
            raw_dic["training"]["task_kwargs"]["pool_dict"]["reg_exempt"] = None
            np.save(os.path.join(out_dir, "config_reg_exempt.npy"), reg_exempt)

        if "NN_W1" in raw_dic["training"]["task_kwargs"]["pool_dict"].keys():
            NN_W1 = raw_dic["training"]["task_kwargs"]["pool_dict"]["NN_W1"].numpy()
            raw_dic["training"]["task_kwargs"]["pool_dict"]["NN_W1"] = None
            np.save(os.path.join(out_dir, "config_NN_W1.npy"), NN_W1)
        if "NN_W2" in raw_dic["training"]["task_kwargs"]["pool_dict"].keys():
            NN_W2 = raw_dic["training"]["task_kwargs"]["pool_dict"]["NN_W2"].numpy()
            raw_dic["training"]["task_kwargs"]["pool_dict"]["NN_W2"] = None
            np.save(os.path.join(out_dir, "config_NN_W2.npy"), NN_W2)
        if "NN_sigma" in raw_dic["training"]["task_kwargs"]["pool_dict"].keys():
            NN_sigma = raw_dic["training"]["task_kwargs"]["pool_dict"]["NN_sigma"].numpy()
            raw_dic["training"]["task_kwargs"]["pool_dict"]["NN_sigma"] = None
            np.save(os.path.join(out_dir, "config_NN_sigma.npy"), NN_sigma)
        if "NN_exempt" in raw_dic["training"]["task_kwargs"]["pool_dict"].keys():
            NN_exempt = raw_dic["training"]["task_kwargs"]["pool_dict"]["NN_exempt"]
            raw_dic["training"]["task_kwargs"]["pool_dict"]["NN_exempt"] = None
            np.save(os.path.join(out_dir, "config_NN_exempt.npy"), NN_exempt)


        if "tree_sep_Ws" in raw_dic["training"]["task_kwargs"]["pool_dict"].keys():
            tree_sep_Ws = raw_dic["training"]["task_kwargs"]["pool_dict"]["tree_sep_Ws"].numpy()
            raw_dic["training"]["task_kwargs"]["pool_dict"]["tree_sep_Ws"] = None
            np.save(os.path.join(out_dir, "config_tree_sep_Ws.npy"), tree_sep_Ws)
        if "tree_target_Leafs" in raw_dic["training"]["task_kwargs"]["pool_dict"].keys():
            tree_target_Leafs = raw_dic["training"]["task_kwargs"]["pool_dict"]["tree_target_Leafs"].numpy()
            raw_dic["training"]["task_kwargs"]["pool_dict"]["tree_target_Leafs"] = None
            np.save(os.path.join(out_dir, "config_tree_target_Leafs.npy"), tree_target_Leafs)
        if "tree_sigma" in raw_dic["training"]["task_kwargs"]["pool_dict"].keys():
            tree_sigma = raw_dic["training"]["task_kwargs"]["pool_dict"]["tree_sigma"].numpy()
            raw_dic["training"]["task_kwargs"]["pool_dict"]["tree_sigma"] = None
            np.save(os.path.join(out_dir, "config_tree_sigma.npy"), tree_sigma)
        if "tree_exempt" in raw_dic["training"]["task_kwargs"]["pool_dict"].keys():
            tree_exempt = raw_dic["training"]["task_kwargs"]["pool_dict"]["tree_exempt"]
            raw_dic["training"]["task_kwargs"]["pool_dict"]["tree_exempt"] = None
            np.save(os.path.join(out_dir, "config_tree_exempt.npy"), tree_exempt)


    if "task_kwargs" in raw_dic["training"].keys() and "mask" in raw_dic["training"]["task_kwargs"].keys():
        mask = raw_dic["training"]["task_kwargs"]["mask"]
        raw_dic["training"]["task_kwargs"]["mask"] = None
        np.save(os.path.join(out_dir, "config_mask.npy"), mask)

    if "task_kwargs" in raw_dic["training"].keys() and "portion_list" in raw_dic["training"]["task_kwargs"].keys():
        portion_list = np.array(raw_dic["training"]["task_kwargs"]["portion_list"])
        raw_dic["training"]["task_kwargs"]["portion_list"] = None
        np.save(os.path.join(out_dir, "config_portion_list.npy"), portion_list)


    with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
                
        yaml.dump(raw_dic, yaml_file, default_flow_style=False)



def config_load_tri(load_dir):
    
    config_path = os.path.join(load_dir, "config.yaml")
    with open(config_path) as fp:  # we don't Quinfig it to avoid inherits
        conf = Munch.fromDict(yaml.safe_load(fp))

    
    mask_path = os.path.join(load_dir, "config_mask.npy")
    mask = np.load(mask_path)
    conf["training"]["task_kwargs"]["mask"] = mask

    portion_path = os.path.join(load_dir, "config_portion_list.npy")
    portion = np.load(portion_path)
    conf["training"]["task_kwargs"]["portion_list"] = portion

    for save_arr in ["reg_w", "reg_sigma", "reg_exempt", 
                     "NN_W1", "NN_W2", "NN_sigma", "NN_exempt",
                     "tree_sep_Ws", "tree_target_Leafs", "tree_sigma", "tree_exempt"]:
        
        conf["training"]["task_kwargs"]["pool_dict"][save_arr] = torch.Tensor(np.load(os.path.join(load_dir, "config_"+save_arr + ".npy")))
    
    for exempt in ["reg_exempt", "NN_exempt", "tree_exempt"]:
        conf["training"]["task_kwargs"]["pool_dict"][exempt] = conf["training"]["task_kwargs"]["pool_dict"][exempt].numpy()

    return conf
            



if __name__ == "__main__":

    raw_args = light_parser().parse_args()

    args = Quinfig(config_path=raw_args.quinine_config_path, schema=schema)

    # print(f"Running with: {args}")

    init_distributed_mode(args.multigpu)

    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")

    if not args.test_run:
        if ( not args.multigpu.distributed ) or (args.multigpu.local_rank == 0):
            run_id = args.training.resume_id
            if run_id is None:
                run_id = str(uuid.uuid4())

            out_dir = os.path.join(args.out_dir, "exp_dir_" + run_id)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            args.out_dir = out_dir

            raw_dic = copy.deepcopy(args.__dict__)
            config_save_special(out_dir, raw_dic)

            # the other two indices, different sigma and whether to exempt from masking

    main(args)
