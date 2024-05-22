import math
import torch.nn as nn
import torch
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from functools import partial
import random
import numpy as np
from utils.misc import seed_all


def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys, mask = None, **kwargs):

    assert len(ys_pred.shape) == 2
    assert len(ys.shape) == 2

    mu = ys_pred

    if mask is not None:
        assert len(mask) >= ys.shape[1]

    full_loss = (ys - ys_pred).square()

    if mask is not None:
        mask = torch.Tensor(mask[:ys.shape[1]])    # .to(full_loss.device)

        return (full_loss[:, torch.where(mask.to(full_loss.device) > 1e-6)[0]]).mean()

    else:
        return full_loss.mean()


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()


def KL_div_muSigma(ys_pred, ys, real_sigs, expand=True, hetero=False):
    if expand:
        if not hetero:
            expanded_sigs = real_sigs * torch.ones(ys.shape).to(ys_pred.device)
        else:
            expanded_sigs = real_sigs.unsqueeze(1).repeat(1, ys.shape[1]).to(ys_pred.device)
    else:
        assert real_sigs.shape == ys.shape
        expanded_sigs = real_sigs

    mu = ys_pred[:, :, 0]
    sigma = nn.Softplus()(ys_pred[:,:, 1])

    p = Normal(ys, expanded_sigs)
    q = Normal(mu, sigma)

    return kl_divergence(p, q)

    


def neg_log_like_muSigma(ys_pred, ys, mask = None, exempt_mode = False, exempt_bools = None):
    assert len(ys_pred.shape) == 3
    assert len(ys.shape) == 2

    mu = ys_pred[:, :, 0]
    sigma = nn.Softplus()(ys_pred[:,:, 1])

    if mask is None:

        loss = nn.GaussianNLLLoss(eps=1e-2)

        return loss(mu, ys, sigma**2)
    
    else:

        assert len(mask) >= ys.shape[1]

        loss = nn.GaussianNLLLoss(eps=1e-02, reduction = 'none')

        full_loss = loss(mu, ys, sigma**2)

        mask = torch.Tensor(mask[:ys.shape[1]])    # .to(full_loss.device)

        if not exempt_mode:

            return (full_loss[:, torch.where(mask.to(full_loss.device) > 1e-6)[0]]).mean()
        
        else:

            full_size_mask = torch.ones((len(exempt_bools), len(mask)))

            full_size_mask[torch.where(torch.Tensor(exempt_bools) < 1e-4)] = mask

            return (full_loss * mask.to(full_loss.device)).mean()







sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, **kwargs):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, **kwargs
):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "sparse_linear_regression": SparseLinearRegression,
        "linear_classification": LinearClassification,
        "noisy_linear_regression": NoisyLinearRegression,
        "quadratic_regression": QuadraticRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "decision_tree": DecisionTree,
        "linear_uncertainty_muSigma": LinearUncertainty_muSigma,
        "tri_mixing_task": tri_mixing_task
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        
        return lambda **args: task_cls(n_dims = n_dims, batch_size=batch_size, pool_dict = pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError


class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, **kwargs):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if ((pool_dict is None) | ("w" not in pool_dict)) and seeds is None:
            self.w_b = torch.randn(self.b_size, self.n_dims, 1)
            if self.n_dims <= 4:
                self.w_b = 5 * self.w_b
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            # indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            indices = random.choices(np.arange(len(pool_dict["w"])), k=batch_size)
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, explicit_seed = None, **kwargs):  # ignore extra args
        if explicit_seed is not None:
            seed_all(explicit_seed)
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric(**kwargs):
        return mean_squared_error


class SparseLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        sparsity=3,
        valid_coords=None,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(SparseLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.sparsity = sparsity
        if valid_coords is None:
            valid_coords = n_dims
        assert valid_coords <= n_dims

        for i, w in enumerate(self.w_b):
            mask = torch.ones(n_dims).bool()
            if seeds is None:
                perm = torch.randperm(valid_coords)
            else:
                generator = torch.Generator()
                generator.manual_seed(seeds[i])
                perm = torch.randperm(valid_coords, generator=generator)
            mask[perm[:sparsity]] = False
            w[mask] = 0

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric(**kwargs):
        return mean_squared_error


class LinearClassification(LinearRegression):
    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric(**kwargs):
        return cross_entropy


class NoisyLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        noise_std_min,
        noise_std_max,
        pool_dict=None,
        seeds=None,
        scale=1,
        noise_std=0,
        renormalize_ys=False,
        special_config = None,
        explicit_seed = None,
        **kwargs
    ):
        """noise_std: standard deviation of noise added to the prediction."""
        super(NoisyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )

        if "w" not in pool_dict:

            if special_config and special_config in ["flip_config1", "flip_config2"]:

                self.w_b = torch.randn(self.b_size, self.n_dims, 1)

                self.w_b[:int(self.b_size/2)] = torch.abs(self.w_b[:int(self.b_size/2)])
                self.w_b[int(self.b_size/2):] = -torch.abs(self.w_b[int(self.b_size/2):])

                if special_config == "flip_config2":

                    self.w_b = 5 * self.w_b

                self.noise_std = torch.Tensor(np.random.uniform(low=0.1, high=0.3, size=self.b_size))

                self.noise_std[int(self.b_size/4):int(self.b_size/2)] = torch.Tensor(np.random.uniform(low=0.5, high=0.7, size=self.b_size))[int(self.b_size/4):int(self.b_size/2)]
                self.noise_std[int(self.b_size/2):int(3*self.b_size/4)] = torch.Tensor(np.random.uniform(low=0.3, high=0.5, size=self.b_size))[int(self.b_size/2):int(3*self.b_size/4)]
                self.noise_std[int(3*self.b_size/4):] = torch.Tensor(np.random.uniform(low=0.7, high=0.9, size=self.b_size))[int(3*self.b_size/4):]

            else:

                self.noise_std_min = noise_std_min
                self.noise_std_max = noise_std_max

                # self.w_b = torch.randn(self.b_size, self.n_dims, 1)
                self.noise_std = torch.Tensor(np.random.uniform(low=noise_std_min, high=noise_std_max, size=self.b_size))
                
            self.hetero = True

            self.exempt_bool = np.zeros(self.b_size)

        else:
            
            if explicit_seed is not None:
                seed_all(explicit_seed)
            indices = random.choices(np.arange(len(pool_dict["w"])), k=batch_size)
            self.w_b = pool_dict["w"][indices]

            if "sigma" in pool_dict: 
                self.noise_std = pool_dict["sigma"][indices]
                self.hetero = True
            else:
                self.noise_std = noise_std
                self.hetero = False

            if "exempt" in pool_dict:
                self.exempt_bool = pool_dict["exempt"][indices]
            else:
                self.exempt_bool = None


        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)

        if not self.hetero:
            ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std
            if self.renormalize_ys:
                ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()
        else:
            assert len(self.noise_std) ==  ys_b.shape[0]

            unsqueeze_noise = self.noise_std.unsqueeze(1).repeat(1, ys_b.shape[1])
            ys_b_noisy = ys_b + torch.randn_like(ys_b) * unsqueeze_noise

        return ys_b_noisy
    
    def eval_with_mu(self, xs_b):
        ys_b = super().evaluate(xs_b)

        if not self.hetero:
            ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std
            if self.renormalize_ys:
                ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()
        else:
            assert len(self.noise_std) ==  ys_b.shape[0]

            unsqueeze_noise = self.noise_std.unsqueeze(1).repeat(1, ys_b.shape[1])
            ys_b_noisy = ys_b + torch.randn_like(ys_b) * unsqueeze_noise

        return ys_b, ys_b_noisy
    
    @staticmethod
    def get_training_metric(mask=None, **kwargs):

        return partial(mean_squared_error, mask = mask)
    
    
class LinearUncertainty_muSigma(NoisyLinearRegression):

    # likelihood as metric


    
    def get_metric(self):
        if self.hetero:
            return partial(KL_div_muSigma, real_sigs = self.noise_std, expand=True, hetero=True)
        else:
            return partial(KL_div_muSigma, real_sigs = self.noise_std, expand=True)
    
    def eval_with_mu_exempt(self, xs_b, explicit_seed = None):
        # ys_b = super().super().evaluate(xs_b)
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]

        if not self.hetero:
            ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std
            if self.renormalize_ys:
                ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()
        else:
            assert len(self.noise_std) ==  ys_b.shape[0]

            if explicit_seed is not None:
                seed_all(explicit_seed)
            rand_part = torch.randn_like(ys_b)
            unsqueeze_noise = self.noise_std.unsqueeze(1).repeat(1, ys_b.shape[1])
            ys_b_noisy = ys_b +  rand_part* unsqueeze_noise

        return ys_b, ys_b_noisy, self.exempt_bool
    
    def eval_with_mu_sigma_exempt(self, xs_b):
        # ys_b = super().super().evaluate(xs_b)
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]

        if not self.hetero:
            ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std
            if self.renormalize_ys:
                ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()
        else:
            assert len(self.noise_std) ==  ys_b.shape[0]

            unsqueeze_noise = self.noise_std.unsqueeze(1).repeat(1, ys_b.shape[1])
            ys_b_noisy = ys_b + torch.randn_like(ys_b) * unsqueeze_noise

        return ys_b, ys_b_noisy, self.noise_std, self.exempt_bool

    @staticmethod
    def get_training_metric(mask=None, exempt_mode = False, exempt_indicator = None, **kwargs):

        if exempt_mode:
            return partial(neg_log_like_muSigma, mask = mask, exempt_mode = True, exempt_bools = exempt_indicator)

        else:
            return partial(neg_log_like_muSigma, mask = mask)




class QuadraticRegression(LinearRegression):
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b_quad = ((xs_b**2) @ w_b)[:, :, 0]
        #         ys_b_quad = ys_b_quad * math.sqrt(self.n_dims) / ys_b_quad.std()
        # Renormalize to Linear Regression Scale
        ys_b_quad = ys_b_quad / math.sqrt(3)
        ys_b_quad = self.scale * ys_b_quad
        return ys_b_quad


class Relu2nnRegression(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        hidden_layer_size=100,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(Relu2nnRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.hidden_layer_size = hidden_layer_size

        if pool_dict is None and seeds is None:
            self.W1 = torch.randn(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.randn(self.b_size, hidden_layer_size, 1)
        elif seeds is not None:
            self.W1 = torch.zeros(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.zeros(self.b_size, hidden_layer_size, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.W1[i] = torch.randn(
                    self.n_dims, hidden_layer_size, generator=generator
                )
                self.W2[i] = torch.randn(hidden_layer_size, 1, generator=generator)
        else:
            assert "W1" in pool_dict and "W2" in pool_dict
            assert len(pool_dict["W1"]) == len(pool_dict["W2"])
            indices = torch.randperm(len(pool_dict["W1"]))[:batch_size]
            self.W1 = pool_dict["W1"][indices]
            self.W2 = pool_dict["W2"][indices]

    def evaluate(self, xs_b):
        W1 = self.W1.to(xs_b.device)
        W2 = self.W2.to(xs_b.device)
        # Renormalize to Linear Regression Scale
        ys_b_nn = (torch.nn.functional.relu(xs_b @ W1) @ W2)[:, :, 0]
        ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        ys_b_nn = self.scale * ys_b_nn
        #         ys_b_nn = ys_b_nn * math.sqrt(self.n_dims) / ys_b_nn.std()
        return ys_b_nn

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        return {
            "W1": torch.randn(num_tasks, n_dims, hidden_layer_size),
            "W2": torch.randn(num_tasks, hidden_layer_size, 1),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric(**kwargs):
        return mean_squared_error


class DecisionTree(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, depth=4):

        super(DecisionTree, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.depth = depth

        if pool_dict is None:

            # We represent the tree using an array (tensor). Root node is at index 0, its 2 children at index 1 and 2...
            # dt_tensor stores the coordinate used at each node of the decision tree.
            # Only indices corresponding to non-leaf nodes are relevant
            self.dt_tensor = torch.randint(
                low=0, high=n_dims, size=(batch_size, 2 ** (depth + 1) - 1)
            )

            # Target value at the leaf nodes.
            # Only indices corresponding to leaf nodes are relevant.
            self.target_tensor = torch.randn(self.dt_tensor.shape)
        elif seeds is not None:
            self.dt_tensor = torch.zeros(batch_size, 2 ** (depth + 1) - 1)
            self.target_tensor = torch.zeros_like(self.dt_tensor)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.dt_tensor[i] = torch.randint(
                    low=0,
                    high=n_dims - 1,
                    size=2 ** (depth + 1) - 1,
                    generator=generator,
                )
                self.target_tensor[i] = torch.randn(
                    self.dt_tensor[i].shape, generator=generator
                )
        else:
            raise NotImplementedError

    def evaluate(self, xs_b):
        dt_tensor = self.dt_tensor.to(xs_b.device)
        target_tensor = self.target_tensor.to(xs_b.device)
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i in range(xs_b.shape[0]):
            xs_bool = xs_b[i] > 0
            # If a single decision tree present, use it for all the xs in the batch.
            if self.b_size == 1:
                dt = dt_tensor[0]
                target = target_tensor[0]
            else:
                dt = dt_tensor[i]
                target = target_tensor[i]

            cur_nodes = torch.zeros(xs_b.shape[1], device=xs_b.device).long()
            for j in range(self.depth):
                cur_coords = dt[cur_nodes]
                cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + cur_decisions

            ys_b[i] = target[cur_nodes]

        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric(**kwargs):
        return mean_squared_error



class SimpleDecisionTree:

    def __init__(
            self, n_dims, depth,
            sep_W, # length depth * n_dims
            target_Leaf, # length 2 ** depth
    ):
        self.n_dims = n_dims
        self.depth = depth
        self.sep_W = sep_W.reshape(depth, -1)
        self.target_Leaf = target_Leaf
        self.aid_tensor = torch.Tensor([2**i for i in range(depth)])

    def passthrough(self, xs):
        sep_W = self.sep_W.to(xs.device)
        target_Leaf = self.target_Leaf.to(xs.device)

        assert xs.shape[1] == self.n_dims

        decision_boundary = xs @ sep_W.T

        decision_tensor =  torch.where(decision_boundary >= 0, torch.tensor(1), torch.tensor(0)).to(xs.device)

        aid_tensor = self.aid_tensor.long().to(xs.device)

        indexes = (decision_tensor @ aid_tensor).long()

        ys = target_Leaf[indexes]

        return ys
    

class LargeDecisionTree:

    def __init__(
            self, n_dims, depth, batch_size, sep_Ws, target_Leafs, noise_std
    ):
        self.n_dims = n_dims
        self.depth = depth
        self.batch_size = batch_size
        assert batch_size == len(sep_Ws)
        self.sep_Ws = sep_Ws
        self.target_Leafs = target_Leafs
        self.noise_std = noise_std

    def eval_with_mu(self, xs_b):

        context_len = xs_b.shape[1]

        ys_b = torch.zeros((self.batch_size, context_len))

        for i in range(self.batch_size):

            little_tree =  SimpleDecisionTree(self.n_dims, self.depth, self.sep_Ws[i], self.target_Leafs[i])

            ys_b[i] = little_tree.passthrough(xs_b[i])

        assert len(self.noise_std) ==  ys_b.shape[0]

        unsqueeze_noise = self.noise_std.unsqueeze(1).repeat(1, ys_b.shape[1])
        ys_b_noisy = ys_b + torch.randn_like(ys_b) * unsqueeze_noise

        return ys_b, ys_b_noisy
    

class tri_mixing_task(Task):
    
    def __init__(self, n_dims, batch_size, pool_dict, hidden_layer_size, depth,
                 portion_list, seeds = None, noise_std = 0., **kwargs):
        super(tri_mixing_task, self).__init__(n_dims, batch_size, pool_dict, seeds)

        self.reg_batSize = int(batch_size* portion_list[0])
        self.NN_batSize = int(batch_size* portion_list[1])
        self.tree_batSize = batch_size - self.reg_batSize - self.NN_batSize

        self.hidden_layer_size = hidden_layer_size
        self.depth = depth
        self.noise_std = noise_std

        assert "reg_w" in pool_dict
        assert "reg_sigma" in pool_dict
            
        reg_indices = random.choices(np.arange(len(pool_dict["reg_w"])), k=self.reg_batSize)
        self.reg_w_b = pool_dict["reg_w"][reg_indices]
        self.reg_noise = pool_dict["reg_sigma"][reg_indices]
        if "reg_exempt" in pool_dict:
            self.reg_exempt_bool = pool_dict["reg_exempt"][reg_indices]
        else:
            self.reg_exempt_bool = None

        assert "NN_W1" in pool_dict
        assert "NN_W2" in pool_dict
        assert "NN_sigma" in pool_dict

        NN_indices = random.choices(np.arange(len(pool_dict["NN_W1"])), k=self.NN_batSize)
        self.NN_W1 = pool_dict["NN_W1"][NN_indices]
        self.NN_W2 = pool_dict["NN_W2"][NN_indices]
        self.NN_noise = pool_dict["NN_sigma"][NN_indices]
        if "NN_exempt" in pool_dict:
            self.NN_exempt_bool = pool_dict["NN_exempt"][NN_indices]
        else:
            self.NN_exempt_bool = None

        assert "tree_sep_Ws" in pool_dict
        assert "tree_target_Leafs" in pool_dict
        assert "tree_sigma" in pool_dict

        tree_indices = random.choices(np.arange(len(pool_dict["tree_sep_Ws"])), k=self.tree_batSize)
        self.tree_sep_Ws = pool_dict["tree_sep_Ws"][tree_indices]
        self.tree_target_Leafs = pool_dict["tree_target_Leafs"][tree_indices]
        self.tree_noise = pool_dict["tree_sigma"][tree_indices]
        if "tree_exempt" in pool_dict:
            self.tree_exempt_bool = pool_dict["tree_exempt"][tree_indices]
        else:
            self.tree_exempt_bool = None


    def eval_with_mu_sigma_exempt(self, xs_b):

        # reg output

        w_b = self.reg_w_b.to(xs_b.device)
        reg_ys_b = (xs_b[:self.reg_batSize] @ w_b)[:, :, 0]

        assert len(self.reg_noise) ==  reg_ys_b.shape[0]

        reg_unsqueeze_noise = self.reg_noise.unsqueeze(1).repeat(1, reg_ys_b.shape[1])
        reg_ys_b_noisy = reg_ys_b + torch.randn_like(reg_ys_b) * reg_unsqueeze_noise


        # NN output

        NN_W1 = self.NN_W1.to(xs_b.device)
        NN_W2 = self.NN_W2.to(xs_b.device)
        # Renormalize to Linear Regression Scale
        NN_ys_b_nn = (torch.nn.functional.relu(xs_b[self.reg_batSize:self.reg_batSize + self.NN_batSize] @ NN_W1) @ NN_W2)[:, :, 0]
        NN_ys_b_nn = NN_ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        
        assert len(self.NN_noise) ==  NN_ys_b_nn.shape[0]

        NN_unsqueeze_noise = self.NN_noise.unsqueeze(1).repeat(1, NN_ys_b_nn.shape[1])
        NN_ys_b_nn_noisy = NN_ys_b_nn + torch.randn_like(NN_ys_b_nn) * NN_unsqueeze_noise


        # decision tree

        context_len = xs_b.shape[1]

        tree_ys_b = torch.zeros((self.tree_batSize, context_len))

        xs_b_treesection = xs_b[self.reg_batSize + self.NN_batSize:]

        for i in range(self.tree_batSize):

            little_tree =  SimpleDecisionTree(self.n_dims, self.depth, self.tree_sep_Ws[i], self.tree_target_Leafs[i])

            tree_ys_b[i] = little_tree.passthrough(xs_b_treesection[i])

        assert len(self.tree_noise) ==  tree_ys_b.shape[0]

        tree_unsqueeze_noise = self.tree_noise.unsqueeze(1).repeat(1, tree_ys_b.shape[1])
        tree_ys_b_noisy = tree_ys_b + torch.randn_like(tree_ys_b) * tree_unsqueeze_noise


        # to sum up

        ys_b = torch.cat((reg_ys_b, NN_ys_b_nn, tree_ys_b), axis = 0)
        ys_b_noisy = torch.cat((reg_ys_b_noisy, NN_ys_b_nn_noisy, tree_ys_b_noisy), axis = 0)

        noise_std = torch.cat((self.reg_noise, self.NN_noise, self.tree_noise), axis = 0)

        exempt_bool = np.concatenate((self.reg_exempt_bool, self.NN_exempt_bool, self.tree_exempt_bool), axis = 0)

        return ys_b, ys_b_noisy, noise_std, exempt_bool
    

    def eval_with_mu_exempt(self, xs_b, explicit_seed=None):

        ys_b, ys_b_noisy, noise_std, exempt_bool = self.eval_with_mu_sigma_exempt(xs_b)

        return ys_b, ys_b_noisy, exempt_bool

    def get_metric(self):
        
        return partial(KL_div_muSigma, real_sigs = self.noise_std, expand=True, hetero=True)
    

    @staticmethod
    def get_training_metric(mask=None, exempt_mode = False, exempt_indicator = None, **kwargs):

        if exempt_mode:
            return partial(neg_log_like_muSigma, mask = mask, exempt_mode = True, exempt_bools = exempt_indicator)

        else:
            return partial(neg_log_like_muSigma, mask = mask)











        





