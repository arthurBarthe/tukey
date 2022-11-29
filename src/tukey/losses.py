"""
Created on Thu Apr  2 23:08:26 2020

@author: arthur
In this module we define custom loss functions. In particular we define
a loss function based on the Gaussian likelihood with two parameters,
mean and precision.
"""
import torch
from torch.nn.modules.loss import _Loss
from scipy.stats import norm
from enum import Enum
from abc import ABC
import numpy as np
from torch.autograd import Function
from torch.nn.functional import softplus


class VarianceMode(Enum):
    variance = 0
    precision = 1


class StudentLoss(_Loss):
    def __init__(self, nu: float = 30, n_target_channels: int = 1):
        super().__init__()
        self.n_target_channels = n_target_channels

    @property
    def n_required_channels(self):
        """Return the number of input channel required per target channel.
        In this case, two, one for the mean, another one for the precision"""
        return 2 * self.n_target_channels

    def pointwise_likelihood(self, input: torch.Tensor, target: torch.Tensor):
        # Temporary fix
        input, nu = input
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        term1 = - torch.lgamma((nu + 1) / 2)
        term2 = 1 / 2 * torch.log(nu) + torch.lgamma(nu / 2)
        term3 = - torch.log(precision)
        temp = (target - mean) * precision
        term4 = (nu + 1) / 2 * torch.log(1 + 1 / nu * temp ** 2)
        return term1 + term2 + term3 + term4

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        lkhs = self.pointwise_likelihood(input, target)
        # Ignore nan values in targets.
        lkhs = lkhs[~torch.isnan(target)]
        return lkhs.mean()

    def predict(self, input: torch.Tensor):
        input, nu = input
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        return mean

    def predict_mean(self, input: torch.Tensor):
        input, nu = input
        """Return the mean of the conditional distribution"""
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        return mean

    @property
    def precision_indices(self):
        return list(range(self.n_target_channels, self.n_required_channels))


class CauchyLoss(_Loss):
    def __init__(self, n_target_channels: int = 1):
        super().__init__()
        self.n_target_channels = n_target_channels

    @property
    def n_required_channels(self):
        """Return the number of input channel required per target channel.
        In this case, two, one for the mean, another one for the precision"""
        return 2 * self.n_target_channels

    def pointwise_likelihood(self, input: torch.Tensor, target: torch.Tensor):
        mean, scale = torch.split(input, self.n_target_channels, dim=1)
        term1 = - torch.log(scale)
        term2 = torch.log((target - mean) ** 2 + scale ** 2)
        return term1 + term2

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        lkhs = self.pointwise_likelihood(input, target)
        # Ignore nan values in targets.
        lkhs = lkhs[~torch.isnan(target)]
        return lkhs.mean()

    def predict(self, input: torch.Tensor):
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        return mean

    def predict_mean(self, input: torch.Tensor):
        """Return the mean of the conditional distribution"""
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        return mean

    @property
    def precision_indices(self):
        return list(range(self.n_target_channels, self.n_required_channels))


class GaussianLoss(_Loss):
    """Class for Gaussian likelihood"""

    def __init__(self, n_target_channels: int = 1, bias: float = 0.,
                 mode=VarianceMode.precision):
        super().__init__()
        self.n_target_channels = n_target_channels
        self.bias = bias
        self.mode = mode

    @property
    def n_required_channels(self):
        """Return the number of input channel required per target channel.
        In this case, two, one for the mean, another one for the precision"""
        return 2 * self.n_target_channels

    @property
    def channel_names(self):
        return ['S_x', 'S_y', 'S_xscale', 'S_yscale']

    @property
    def precision_indices(self):
        return list(range(self.n_target_channels, self.n_required_channels))

    def pointwise_likelihood(self, input: torch.Tensor, target: torch.Tensor):
        # Split the target into mean (first half of channels) and scale
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        if not torch.all(precision > 0):
            raise ValueError('Got a non-positive variance value. \
                             Pre-processed variance tensor was: \
                                 {}'.format(torch.min(precision)))
        if self.mode is VarianceMode.precision:
            term1 = - torch.log(precision)
            term2 = 1 / 2 * (target - (mean + self.bias)) ** 2 * precision ** 2
        elif self.mode is VarianceMode.variance:
            term1 = torch.log(precision)
            term2 = 1 / 2 * (target - (mean + self.bias)) ** 2 / precision ** 2
        return term1 + term2

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        precision = self._transform_precision(precision)
        input = torch.cat((mean, precision), dim=1)
        lkhs = self.pointwise_likelihood(input, target)
        # Ignore nan values in targets.
        lkhs = lkhs[~torch.isnan(target)]
        return lkhs.mean()

    def predict(self, input: torch.Tensor):
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        precision = self._transform_precision(precision)
        return mean + self.bias, precision

    def predict_mean(self, input: torch.Tensor):
        """Return the mean of the conditional distribution"""
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        return mean + self.bias

    def residuals(self, input, target):
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        precision = self._transform_precision(precision)
        return (target - mean) * precision

    def _transform_precision(self, precision):
        return softplus(precision)

    def sample(self, params: torch.tensor, z: torch.tensor = None):
        """
        Sample from i.i.d. Gaussian distributions with parameters specified by the passed params (although they still
        need to be transformed.

        Parameters
        ----------
        params
            parameters of the Gaussian distributions with shape (N, 2 * C, H, W)
        z
            standardized normal, can be used when we want to add spatio-temporal correlation
        """
        mean, precision = torch.split(params, dim=1)
        precision = self._transform_precision(precision)
        if z is None:
            z = torch.randn_like(mean)
        return mean + 1 / precision * z




class Tuckey_g_h_inverse(Function):

    @staticmethod
    def tuckey_g_h(z, g, h):
        out = 1 / g * torch.expm1(g * z) * torch.exp(h * z ** 2 / 2)
        return out

    @staticmethod
    def forward(ctx, z_tilda, g, h):
        """New inverse method based on dichotomial algorithm. Finds true
        inverse up to numerical precision"""
        g[g == 0.] = torch.finfo().eps
        min_ = -10
        max_ = 10
        min_ = torch.ones_like(z_tilda) * min_
        max_ = torch.ones_like(z_tilda) * max_
        middle = min_
        for i in range(30):
            old_middle = middle
            middle = (min_ + max_) / 2.
            if torch.all(torch.abs(middle - old_middle) <= torch.finfo(
                    middle.dtype).eps):
                break
            value = Tuckey_g_h_inverse.tuckey_g_h(middle, g, h)
            max_[z_tilda < value] = middle[z_tilda < value]
            min_[z_tilda > value] = middle[z_tilda > value]
        middle = (min_ + max_) / 2
        middle[torch.isnan(z_tilda)] = np.nan
        ctx.save_for_backward(middle, g, h)
        return middle

    @staticmethod
    def old_forward(ctx, z_tilda, g, h):
        g[g == 0.] = torch.finfo().eps
        nodes = torch.linspace(-4, 4, 1000, device=z_tilda.device)
        nodes = nodes.reshape([1, ] * z_tilda.ndim + [1000, ])
        new_g = g.unsqueeze(-1)
        new_h = h.unsqueeze(-1)
        init_shape = z_tilda.shape
        z_tilda = z_tilda.unsqueeze(-1)
        node_values = Tuckey_g_h_inverse.tuckey_g_h(nodes, new_g, new_h)
        if torch.any(node_values.isnan()):
            indices = torch.nonzero(node_values.isnan())
            first_index = list(indices[0])
            print(first_index)
            print(g[first_index[:-1]])
            print(h[first_index[:-1]])
            print(nodes.flatten()[first_index[-1]])
            raise RuntimeError('Got nan in node values.')

        assert not torch.any(node_values.isinf()), "Got inf in node values"
        i_node = torch.argmax((z_tilda <= node_values) * 1., dim=-1,
                              keepdim=True)
        i_node[z_tilda > node_values[..., -1:]] = 999
        nodes = nodes.flatten()
        z = nodes[i_node]
        z = z.reshape(init_shape)
        ctx.save_for_backward(z, g, h)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        z, g, h = ctx.saved_tensors
        if grad_output is None:
            return None
        Tuckey_g_h_inverse.assert_finite(grad_output)
        d_input = 1 / Tuckey_g_h_inverse.d_tau_d_z(z, g, h)
        d_g = - Tuckey_g_h_inverse.d_tau_d_g(z, g, h) * d_input
        d_h = - Tuckey_g_h_inverse.d_tau_d_h(z, g, h) * d_input
        Tuckey_g_h_inverse.assert_finite(d_input, d_g, d_h)
        return d_input * grad_output, d_g * grad_output, d_h * grad_output

    @staticmethod
    def d_tau_d_g(z, g, h):
        out = - 1 / g * Tuckey_g_h_inverse.tuckey_g_h(z, g, h)
        out = out + 1 / g * z * torch.exp(g * z + 1 / 2 * h * z ** 2)
        Tuckey_g_h_inverse.assert_finite(out)
        return out

    @staticmethod
    def d_tau_d_h(z, g, h):
        return 1 / 2 * z ** 2 * Tuckey_g_h_inverse.tuckey_g_h(z, g, h)

    @staticmethod
    def d_tau_d_z(z, g, h):
        out = torch.exp(g * z + h * z ** 2 / 2)
        out = out + h * z * Tuckey_g_h_inverse.tuckey_g_h(z, g, h)
        return out

    @staticmethod
    def assert_finite(*args):
        for arg in args:
            assert torch.all(torch.isfinite(arg))


class TuckeyGandHloss(_Loss):
    def __init__(self, n_target_channels: int = 2):
        super().__init__()
        self.n_target_channels = n_target_channels
        self.inverse_tuckey = Tuckey_g_h_inverse()

    @property
    def n_required_channels(self):
        """Return the number of required channels for the input. For each
        component of the target, 4 input channels are required: 1 for the
        constant, one for the scale, two for the Tuckey g and h parameters"""
        return self.n_target_channels * 4

    def forward(self, input, target):
        epsilon, beta, g, h = torch.split(input, self.n_target_channels, dim=1)
        beta = self._transform_beta(beta)
        g, h = self._transform_g_h(g, h)
        z_tilda = (target - epsilon) * beta
        z = self.inverse_tuckey.apply(z_tilda, g, h)
        assert not torch.any(z.isnan()), "Got nan values in the inversion"
        assert not torch.any(z.isinf()), "Got inf values in the inversion"
        for_log = (h / g * z * torch.expm1(g * z) + torch.exp(g * z))
        assert not torch.any(for_log.isnan()), "Got nan values in for_log"
        assert not torch.any(for_log.isinf()), "Got inf values in for inf"
        lkh = torch.log(for_log)
        lkh = lkh + (h + 1) / 2 * z ** 2
        lkh = lkh - torch.log(beta)
        lkh = lkh.mean()
        return lkh

    @property
    def channel_names(self):
        return ['epsilon_Sx', 'epsilon_Sy', 'beta_Sx', 'beta_Sy',
                'g_Sx', 'g_Sy', 'h_Sx', 'h_Sy']

    def predict(self, input):
        epsilon, beta, g, h = torch.split(input, self.n_target_channels, dim=1)
        beta = self._transform_beta(beta)
        g, h = self._transform_g_h(g, h)
        return epsilon, beta, g, h

    def predict_ci(self, input, alpha=0.95):
        epsilon, beta, g, h = torch.split(input, self.n_target_channels, dim=1)
        beta = self._transform_beta(beta)
        g, h = self._transform_g_h(g, h)
        lb = norm.ppf((1 - alpha) / 2)
        ub = -lb
        lb = epsilon + self.inverse_tuckey.tuckey_g_h(lb, g, h) / beta
        ub = epsilon + self.inverse_tuckey.tuckey_g_h(ub, g, h) / beta
        return lb, ub

    def residuals(self, input, target):
        epsilon, beta, g, h = torch.split(input, self.n_target_channels, dim=1)
        beta = self._transform_beta(beta)
        g, h = self._transform_g_h(g, h)
        z_tilda = (target - epsilon) * beta
        z = self.inverse_tuckey.apply(z_tilda, g, h)
        return z

    def _transform_g_h(self, g, h):
        g = (torch.sigmoid(g) - 0.5) * 2
        # works well enough
        # h = torch.nn.functional.softplus(h)
        h = torch.exp(-h)
        return g, h

    def _transform_beta(self, beta):
        return softplus(beta)

    def sample(self, params: torch.tensor, z: torch.tensor = None):
        """
        Sample from i.i.d. Gaussian distributions with parameters specified by the passed params (although they still
        need to be transformed.

        Parameters
        ----------
        params
            parameters of the Gaussian distributions with shape (N, 2 * C, H, W)
        z
            standardized normal, can be used when we want to add spatio-temporal correlation
        """
        epsilon, beta, g, h = torch.split(input, self.n_target_channels, dim=1)
        beta = self._transform_beta(beta)
        g, h = self._transform_g_h(g, h)
        if z is None:
            z = torch.randn_like(epsilon)
        return epsilon + 1 / beta * self.inverse_tuckey.tuckey_g_h(z, g, h)


if __name__ == '__main__':
    input = np.random.randn(4, )
    input = torch.reshape(torch.tensor(input), (1, 4, 1, 1))
    input.requires_grad = True
    target = (np.random.randn(1, ) - 0.5)
    target = torch.reshape(torch.tensor(target), (1, 1, 1, 1))
    tgh = TuckeyGandHloss(n_target_channels=1)
    z = tgh(input, target)
    print(z)

    from torch.autograd import gradcheck

    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    print('start test')
    test2 = gradcheck(tgh.forward, (input, target), eps=1e-4, atol=0.01)
    print(test2)

