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
import wandb
from utils.manual_solvers import w_sig_NGgenerator, w_sig_NGposterior, w_sig_ridge
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
    # parser.add_argument('--pool_size', type=int, default=0)
    # parser.add_argument('--mask_scheme', type=str)
    # parser.add_argument('--mask_scheme', type=str)

    return parser




if __name__ == "__main__":

    raw_args = mass_parser().parse_args()

    mask_scheme = "nomask"

    template_args = Quinfig(config_path=raw_args.quinine_config_path, schema=schema)

    template_args.training.task_kwargs["exempt_mode"] = True

    pool_size = 16384

    init_distributed_mode(template_args.multigpu)

    x_dim = template_args.model.n_dims

    mu0 = np.ones(x_dim)

    ## gamma(a, b), mean = a/b, variance = a/b^2
    ## when training, we let a/b=1 and b relatively large
    b0 = 20
    a0 = 20
    
    ws, sigs = w_sig_NGgenerator(a0, b0, mu0, x_dim, pool_size)



    pool_of_w = torch.Tensor(np.expand_dims(ws, axis=-1))
    
    pool_of_sigma = torch.Tensor(sigs)
    
    pool_of_exempt = np.zeros(pool_size)

    args = copy.deepcopy(template_args)

    args.training.task_kwargs["pool_dict"] = {
        "w": pool_of_w,
        "sigma": pool_of_sigma,
        "exempt": pool_of_exempt
    }

    # -------------------add mask---------------------------------
    assert mask_scheme in ["leave045", "mask045", "nomask"]

    max_sampe_len = args.training.curriculum.points.end + 1

    mask = np.zeros(max_sampe_len)

    if mask_scheme == "leave045":
        mask[:45] =  1
    elif mask_scheme == "mask045":
        mask[45:] = 1
    elif mask_scheme == "nomask":
        mask[:] = 1

    args.training.task_kwargs["mask"] = mask
    

    # ------------------don't forget------------------------------
    exp_name = args.wandb.name
    args.wandb.name = exp_name
    # ------------------don't forget------------------------------

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

        groups = {
            "low_sig": (80, 20),
            "original": (20, 20),
            "large_sig": (100, 400)
        }

        output = {}

        for param_config in groups.keys():


            a_conf, b_conf = groups[param_config]

            step_group = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

            N_sample = 200

            data_sampler = get_data_sampler(args.training.data, n_dims=args.model.n_dims)

            # ----------------------------------------construct task sampler---------------

            new_ws, new_sigs = w_sig_NGgenerator(a_conf, b_conf, mu0, x_dim, N_sample)

            new_pool_of_w = torch.Tensor(np.expand_dims(new_ws, axis=-1))
        
            new_pool_of_sigma = torch.Tensor(new_sigs)
            

            # manual evaluate        

            eval_xs = data_sampler.sample_xs(
                args.training.curriculum.points.end,
                N_sample,
                args.training.curriculum.dims.end,
            )

            eval_mus = (eval_xs @ new_pool_of_w)[:, :, 0]

            unsqueeze_noise = new_pool_of_sigma.unsqueeze(1).repeat(1, eval_mus.shape[1])
            eval_ys = eval_mus + torch.randn_like(eval_mus) * unsqueeze_noise


            with torch.no_grad():
                eval_pred = eval_model(eval_xs, eval_ys)

            # start the recording
                
            eval_predmu = eval_pred[:, :, 0]
            eval_predsigma = eval_pred[:, :, 1]

            def softplus(arr):
                return np.log(1+np.exp(arr))
            
            eval_mu_diff = np.abs(eval_predmu.cpu().numpy()-eval_mus.cpu().numpy()).mean(axis=0)


            real_eval_pred_sigma = softplus(eval_predsigma.cpu().numpy())

            eval_full_sigma  = np.tile(new_sigs.reshape(-1,1), (1, eval_predsigma.shape[1]))

            eval_sigma_diff = np.abs(real_eval_pred_sigma-eval_full_sigma).mean(axis=0)


            posterior_mu = np.zeros(eval_predmu.shape)
            posterior_sigma = np.zeros(eval_predmu.shape)
            ridge_mu = np.zeros(eval_predmu.shape)
            ridge_sigma = np.zeros(eval_predmu.shape)


            for selected_index in range(len(eval_predmu)):

                estimated_ws, estimated_sigs = w_sig_NGposterior(eval_xs.numpy()[selected_index], eval_ys.numpy()[selected_index], a0, b0, mu0)

                posterior_mu[selected_index, 1:] = (np.expand_dims(eval_xs.numpy()[selected_index, 1:], axis=1) @ np.expand_dims(estimated_ws[:-1], axis=-1))[:, 0, 0]
                posterior_sigma[selected_index, 1:] = estimated_sigs[:-1]


                ridge_ws, ridge_sigs = w_sig_ridge(eval_xs.numpy()[selected_index], eval_ys.numpy()[selected_index])
                
                ridge_mu[selected_index, 1:] = (np.expand_dims(eval_xs.numpy()[selected_index, 1:], axis=1) @ np.expand_dims(ridge_ws[:-1], axis=-1))[:, 0, 0]
                ridge_sigma[selected_index, 1:] = ridge_sigs[:-1]


            mu_real_diff = eval_mu_diff
            sig_real_diff = eval_sigma_diff

            mu_posterior_diff = np.abs(eval_predmu.cpu().numpy() - posterior_mu).mean(axis=0)
            mu_ridge_diff = np.abs(eval_predmu.cpu().numpy() - ridge_mu).mean(axis=0)
            sig_posterior_diff = np.abs(real_eval_pred_sigma - posterior_sigma).mean(axis=0)
            sig_ridge_diff = np.abs(real_eval_pred_sigma - ridge_sigma).mean(axis=0)

            mu_posterior_real_diff = np.abs(posterior_mu - eval_mus.cpu().numpy()).mean(axis=0)
            mu_ridge_real_diff = np.abs(ridge_mu - eval_mus.cpu().numpy()).mean(axis=0)
            sig_posterior_real_diff = np.abs(posterior_sigma - eval_full_sigma).mean(axis=0)
            sig_ridge_real_diff = np.abs(ridge_sigma - eval_full_sigma).mean(axis=0)



            

            for step in step_group:

                # AbsDev include
                # mu_to_real
                # mu_to_posterior
                # mu_to_posterior / posterior_to_real
                # mu_to_ridge
                # mu_to_ridge / ridge_to_real

                # sig_to_real
                # sig_to_posterior
                # sig_to_posterior / posterior_to_real
                # sig_to_ridge
                # sig_to_ridge / ridge_to_real


                output["mu_real_diff" + "_noiseLevel_" + param_config + "_at_" + str(step)] = mu_real_diff[step]
                output["sig_real_diff"+ "_noiseLevel_" + param_config  + "_at_" + str(step)] = sig_real_diff[step]

                output["mu_posterior_diff"+ "_noiseLevel_" + param_config  + "_at_" + str(step)] = mu_posterior_diff[step]
                output["mu_ridge_diff"+ "_noiseLevel_" + param_config  + "_at_" + str(step)] = mu_ridge_diff[step]
                output["sig_posterior_diff"+ "_noiseLevel_" + param_config  + "_at_" + str(step)] = sig_posterior_diff[step]
                output["sig_ridge_diff"+ "_noiseLevel_" + param_config  + "_at_" + str(step)] = sig_ridge_diff[step]

                output["mu_posterior_real_diff"+ "_noiseLevel_" + param_config  + "_at_" + str(step)] = mu_posterior_real_diff[step]
                output["mu_ridge_real_diff"+ "_noiseLevel_" + param_config  + "_at_" + str(step)] = mu_ridge_real_diff[step]
                output["sig_posterior_real_diff"+ "_noiseLevel_" + param_config  + "_at_" + str(step)] = sig_posterior_real_diff[step]
                output["sig_ridge_real_diff"+ "_noiseLevel_" + param_config  + "_at_" + str(step)] = sig_ridge_real_diff[step]

        return output


    main(args, use_wandb= True, panel_plot=panel_plot_func) 


        

