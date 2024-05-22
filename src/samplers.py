import math

import torch
from utils.misc import seed_all


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "varsig": VarSigSampler
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None, explicit_seed = None):
        if explicit_seed is not None:
            seed_all(explicit_seed)
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b


class VarSigSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None, explicit_seed = None,
                  upper_limit_vec=None, lower_limit_vec=None):
        if explicit_seed is not None:
            seed_all(explicit_seed)
        assert seeds is None

        xs_b = torch.randn(b_size, n_points, self.n_dims)
        if upper_limit_vec is None:
            scale = torch.sqrt(2 * torch.rand(b_size, self.n_dims)).unsqueeze(1).repeat(1, n_points, 1)
        else:
            assert len(upper_limit_vec) == self.n_dims
            if not isinstance(upper_limit_vec, torch.Tensor):
                # Convert array to tensor
                upper_limit_vec = torch.tensor(upper_limit_vec)
            
            if lower_limit_vec is None:
                lower_limit_vec = torch.zeros(len(upper_limit_vec))
            elif not isinstance(lower_limit_vec, torch.Tensor):
                lower_limit_vec = torch.Tensor(lower_limit_vec)

            scale = torch.sqrt(lower_limit_vec.unsqueeze(0).repeat(b_size, 1) +  (upper_limit_vec-lower_limit_vec).unsqueeze(0).repeat(b_size, 1) * torch.rand(b_size, self.n_dims)).unsqueeze(1).repeat(1, n_points, 1)
        xs_b = xs_b * scale

        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b