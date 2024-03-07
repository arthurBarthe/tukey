"""
In this module we define classes that allow to use a multimodal distribution for the loss function.
For instance, instead of training a neural network to ouptput the two parameters of a Gaussian
distribution, the neural network may output the parameters of two Gaussian distributions.
"""

import torch
from torch.nn.modules.loss import _Loss
from torch.nn.functional import softplus


class MultimodalLoss(_Loss):
    """General class for a multimodal loss. Each location on
    each channel can choose its mode independently."""

    def __init__(self, n_modes, n_target_channels, base_loss_cls,
                 base_loss_params=[], share_mode='C'):
        super().__init__()
        self.n_modes = n_modes
        self.n_target_channels = n_target_channels
        self.target_names = ['target' + str(i) for i in range(
                n_target_channels)]
        self.losses = []
        for i in range(n_modes):
            if i < len(base_loss_params):
                params = base_loss_params[i]
                self.losses.append(base_loss_cls(n_target_channels, **params))
            else:
                self.losses.append(base_loss_cls(n_target_channels))
        self.share_mode = share_mode

    @property
    def target_names(self):
        return self._target_names

    @target_names.setter
    def target_names(self, value):
        assert len(value) == self.n_target_channels
        self._target_names = value

    @property
    def n_required_channels(self):
        if self.share_mode == 'C':
            return sum(self.splits)

    @property
    def channel_names(self):
        """Automatically assigns names to output channels depending on the
        target names. For now not really implemented"""
        return [str(i) for i in range(self.n_required_channels)]

    @property
    def precision_indices(self):
        indices = []
        for i, loss in enumerate(self.losses):
            sub_indices = loss.precision_indices
            for j in range(len(sub_indices)):
                sub_indices[j] += self.n_modes * self.n_target_channels + i * loss.n_required_channels
            indices.extend(sub_indices)
        return indices

    @property
    def splits(self):
        """Return how to split the input to recover the different parts:
            - probabilities of the modes
            - quantities definining each mode
        """
        return ([self.n_modes, ] * self.n_target_channels
                + [loss.n_required_channels for loss in self.losses])

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        splits = torch.split(input, self.splits, dim=1)
        probas, inputs = (splits[:self.n_target_channels],
                          splits[self.n_target_channels:])
        probas = [torch.softmax(proba, dim=1) for proba in probas]
        losses_values = []
        for i, (loss, input) in enumerate(zip(self.losses, inputs)):
            proba_i = torch.stack([proba[:, i, ...] for proba in probas], dim=1)
            loss_i = - loss.pointwise_likelihood(input, target)
            #print(loss_i)
            loss_i = loss_i + torch.log(proba_i)
            losses_values.append(loss_i)
        loss = torch.stack(losses_values, dim=2)
        final_loss = -torch.logsumexp(loss, dim=2)
        final_loss = final_loss.mean()
        return final_loss

    def predict(self, input: torch.Tensor):
        splits = torch.split(input, self.splits, dim=1)
        probas, inputs = (splits[:self.n_target_channels],
                          splits[self.n_target_channels:])
        probas = [torch.softmax(proba, dim=1) for proba in probas]
        predictions = [loss.predict(input) for loss, input in
                       zip(self.losses, inputs)]
        return probas, predictions


class BimodalGaussianLoss(MultimodalLoss):
    """Class for a bimodal Gaussian loss."""

    def __init__(self, n_target_channels: int):
        super().__init__(2, n_target_channels,
                         base_loss_cls=HeteroskedasticGaussianLossV2)


class BimodalStudentLoss(MultimodalLoss):
    def __init__(self, n_target_channels: int):
        super().__init__(2, n_target_channels, base_loss_cls=StudentLoss)


class TrimodalGaussianLoss(MultimodalLoss):
    def __init__(self, n_target_channels: int):
        super().__init__(3, n_target_channels,
                         base_loss_cls=HeteroskedasticGaussianLossV2)


class PentamodalGaussianLoss(MultimodalLoss):
    def __init__(self, n_target_channels: int):
        super().__init__(5, n_target_channels,
                         base_loss_cls=HeteroskedasticGaussianLossV2)

