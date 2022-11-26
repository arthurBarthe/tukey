"""
In this file we define the neural networks used in the application to subgrid parameterization of advection.
"""

import torch
from torch.nn import Sequential, ReLU, Module
from pytorch_wavelets import DWTForward, DWTInverse
from torch.nn.functional import interpolate


class ResNet(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        n_in = module[0].in_channels
        n_out = module[0].out_channels
        self.skip = torch.nn.Conv2d(n_in, n_out, 1)

    def forward(self, inputs):
        return self.module(inputs) + self.skip(inputs)


class FullyCNN(Sequential):

    def __init__(self, n_in_channels: int = 2, n_out_channels: int = 4,
                 padding='same', batch_norm=True, size: int = 3):
        if padding is None:
            padding_5 = 0
            padding_3 = 0
        elif padding == 'same':
            padding_5 = 2
            padding_3 = 1
        else:
            raise ValueError('Unknow value for padding parameter.')
        self.n_in_channels = n_in_channels
        self.batch_norm = batch_norm
        conv1 = torch.nn.Conv2d(n_in_channels, 32, size, padding=padding, padding_mode='circular')
        block1 = self._make_subblock(conv1)
        conv2 = torch.nn.Conv2d(8, 8, size, padding=padding, padding_mode='circular')
        block2 = self._make_subblock(conv2)
        conv3 = torch.nn.Conv2d(8, 8, size, padding=padding, padding_mode='circular')
        block3 = self._make_subblock(conv3)
        conv4 = torch.nn.Conv2d(8, 8, size, padding=padding, padding_mode='circular')
        block4 = self._make_subblock(conv4)
        conv5 = torch.nn.Conv2d(8, 8, size, padding=padding, padding_mode='circular')
        block5 = self._make_subblock(conv5)
        conv8 = torch.nn.Conv2d(32, n_out_channels, size, padding=padding, padding_mode='circular')
        Sequential.__init__(self, block1, conv8)
        #Sequential.__init__(self, block1, block2, block3, block4, block5, conv8)

    def forward(self, x):
        return super().forward(x)
        #return super().forward(x)[:, :, 10:-10, 10:-10]

    def _make_subblock(self, conv):
        subbloc = [conv, ReLU()]
        if self.batch_norm:
            subbloc.append(torch.nn.BatchNorm2d(conv.out_channels))
        return ResNet(Sequential(*subbloc))
########################################################################################################################


class WaveletUnet(Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 4, levels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.levels = levels
        self.xfm = DWTForward(J=self.levels, mode='periodization', wave='db3')
        self.ifm = DWTInverse(mode='periodization', wave='db3')
        self.fcnn_app = FullyCNN(in_channels, out_channels)
        self.fcnn_detail = FullyCNN(in_channels * 3, out_channels * 3)
        self.fcnn_combine = FullyCNN(out_channels, out_channels)

    def forward(self, input: torch.Tensor):
        # input[N, C, H, W]
        # we apply the DWT: app[N, C, H2, W2], detail[N, C, 3, H2, W2]
        app, detail = self.xfm(input)
        # We concatenate the three directions along the channel dimension: detail[N, 3*C, H2, W2]
        detail = [torch.cat([d[:, :, i, ...] for i in range(3)], dim=1) for d in detail]
        # we pass app through a conv net: out_app[N, C_out, H2, W2]
        out_app = self.fcnn_app(app)
        # we pass detail through a conv net: out_detail[N, 3 * C_out, H2, W2]
        out_detail = self.fcnn_detail(detail[0])
        # we apply a third FCNN to combine
        in_combine = torch.cat((out_app, out_detail), dim=1)
        out = self.fcnn_combine(in_combine)
        out_app, out_detail = out[:, :self.out_channels], out[:, self.out_channels:]
        # we transform to allow IDWT
        out_detail = torch.stack(torch.split(out_detail, self.out_channels, dim=1), dim=2)
        # we carry out an IDWT
        out = self.ifm((out_app, (out_detail,)))
        return out[:, :, 10: -10, 10: -10]


class WaveletUnet2(Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 4, levels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.levels = levels
        self.xfm = DWTForward(J=1, mode='periodization', wave='haar')
        self.ifm = DWTInverse(mode='periodization', wave='haar')
        self.fcnn_app = FullyCNN(in_channels, out_channels)
        self.fcnn_detail = FullyCNN(in_channels * 4, out_channels * 3)
        self.fcnn_detail2 = FullyCNN(in_channels * 4, out_channels * 3)
        self.fcnn_detail3 = FullyCNN(in_channels * 4, out_channels * 3)
        # self.fcnn_detail2 = self.fcnn_detail
        self.fcnn_combine0 = FullyCNN(out_channels, out_channels, size=3)
        self.fcnn_combine1 = FullyCNN(out_channels, out_channels, size=3)
        self.fcnn_combine2 = FullyCNN(out_channels, out_channels, size=3)

    def forward(self, input: torch.Tensor):
        # input[N, C, H, W]
        # we apply the DWT: app[N, C, H2, W2], detail[N, C, 3, H2, W2]
        app, detail = self.xfm(input)
        # We concatenate the three directions along the channel dimension: detail[N, 3*C, H2, W2]
        detail = [torch.cat([d[:, :, i, ...] for i in range(3)], dim=1) for d in detail]
        # we pass detail through a conv net: out_detail[N, 3 * C_out, H2, W2]
        out_detail = self.fcnn_detail(torch.cat((app, detail[0]), dim=1))
        # we transform to allow IDWT
        out_detail1 = torch.stack(torch.split(out_detail, self.out_channels, dim=1), dim=2)

        # next level of DWT--------------------
        app, detail = self.xfm(app)
        detail = [torch.cat([d[:, :, i, ...] for i in range(3)], dim=1) for d in detail]
        # we pass detail through a conv net: out_detail[N, 3 * C_out, H2, W2]
        out_detail = self.fcnn_detail2(torch.cat((app, detail[0]), dim=1))
        # we transform to allow IDWT
        out_detail2 = torch.stack(torch.split(out_detail, self.out_channels, dim=1), dim=2)

        # next level of DWT--------------------
        app, detail = self.xfm(app)
        detail = [torch.cat([d[:, :, i, ...] for i in range(3)], dim=1) for d in detail]
        # we pass detail through a conv net: out_detail[N, 3 * C_out, H2, W2]
        out_detail = self.fcnn_detail3(torch.cat((app, detail[0]), dim=1))
        # we transform to allow IDWT
        out_detail3 = torch.stack(torch.split(out_detail, self.out_channels, dim=1), dim=2)

        out_app = self.fcnn_app(app)

        # we carry out an IDWT
        out2 = self.ifm((out_app, (out_detail3,)))
        # we combine
        out2 = self.fcnn_combine2(out2)

        # we carry out an IDWT
        out1 = self.ifm((out2, (out_detail2,)))
        # we combine
        out1 = self.fcnn_combine1(out1)
        # we carry out an IDWT
        out = self.ifm((out1, (out_detail1,)))
        # we combine
        out = self.fcnn_combine0(out)
        return out[:, :, 10: -10, 10: -10]


class WaveletsNN(Module):
    def __init__(self, n_in_channels: int = 2, n_out_channels: int = 4, device='cpu'):
        super().__init__()
        self.fcnn_app = FullyCNN(n_in_channels * 3 * 3 + 2, n_out_channels, padding='same')
        self.fcnn_detail1 = FullyCNN(n_in_channels * 3 * 3 + 2, n_out_channels * 3, padding='same')
        self.fcnn_detail2 = FullyCNN(n_in_channels * 3 * 3 + 2, n_out_channels * 3, padding='same')
        self.fcnn_detail3 = FullyCNN(n_in_channels * 3 * 3 + 2, n_out_channels * 3, padding='same')
        # self.fcnn_detail4 = FullyCNN(n_in_channels * 3 + 2, n_out_channels * 3, padding='same')
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.xfm = DWTForward(J=3, mode='periodization', wave='haar')  # Accepts all wave types available to PyWavelets
        self.ifm = DWTInverse(mode='periodization', wave='haar')
        if device == 'cuda':
            self.xfm = self.xfm.cuda()
            self.ifm = self.ifm.cuda()

    def forward(self, input):
        # Wavelet transform. App is the remaining coarse scale stuff.
        app, detail = self.xfm(input)
        # detail1 is finest scale, detail3 is coarser scale.
        # We concatenate the three directions along the channel dimension
        detail = [torch.cat([d[:, :, i, ...] for i in range(3)], dim=1) for d in detail]
        detail1, detail2, detail3 = detail
        # We do not allow coarse scale stuff in the output
        # out_app = torch.zeros((app.shape[0], self.n_out_channels, *app.shape[-2:])).to(device=DEVICE)
        out_app = self.fcnn_app(self._shape_like(app, [detail1, detail2, detail3]))
        # Each scale is obtained by applying a FCNN using the input at that scale and the coarser scale
        # out_detail1 = self.fcnn_detail1(torch.cat((interpolate(detail2, detail1.shape[-2:]), detail1), dim=1))
        # out_detail2 = self.fcnn_detail2(torch.cat((interpolate(detail3, detail2.shape[-2:]), detail2), dim=1))
        # out_detail3 = self.fcnn_detail3(torch.cat((interpolate(app, detail3.shape[-2:]), detail3), dim=1))
        # out_detail3 = self.fcnn_detail3(torch.cat((interpolate(detail4, detail3.shape[-2:]), detail3), dim=1))
        # out_detail4 = self.fcnn_detail4(torch.cat((app, detail4), dim=1))
        out_detail1 = self.fcnn_detail1(self._shape_like(detail1, [detail2, detail3, app]))
        out_detail2 = self.fcnn_detail2(self._shape_like(detail2, [detail1, detail3, app]))
        out_detail3 = self.fcnn_detail3(self._shape_like(detail3, [detail1, detail2, app]))

        # The 4 first layers of the output give the horizontal coefficients for the 4 layers
        # The 4 next layers of the output give the vertical coefficients for the 4 layers
        # The 4 next layers of the output give the diagonal coefficients for the 4 layers
        out_detail1 = torch.stack(torch.split(out_detail1, self.n_out_channels, dim=1), dim=2)
        out_detail2 = torch.stack(torch.split(out_detail2, self.n_out_channels, dim=1), dim=2)
        out_detail3 = torch.stack(torch.split(out_detail3, self.n_out_channels, dim=1), dim=2)
        # out_detail4 = torch.stack(torch.split(out_detail4, self.n_out_channels, dim=1), dim=2)
        # out_detail1 = torch.reshape(out_detail1, (detail1.shape[0], self.n_out_channels, 3, *detail1.shape[-2:]))
        # out_detail2 = torch.reshape(out_detail2, (detail2.shape[0], self.n_out_channels, 3, *detail2.shape[-2:]))
        # out_detail3 = torch.reshape(out_detail3, (detail3.shape[0], self.n_out_channels, 3, *detail3.shape[-2:]))
        # We add direct connections
        out_app = out_app
        out_detail1 = out_detail1
        out_detail2 = out_detail2
        out_detail3 = out_detail3
        out = self.ifm((out_app, (out_detail1, out_detail2, out_detail3)))
        return out[:, :, 10:-10, 10:-10]

    def _shape_like(self, a: torch.Tensor, others: list):
        spatial_dimensions = a.shape[-2:]
        new_others = [interpolate(t, spatial_dimensions, mode='bilinear') for t in others]
        return torch.cat((a, *new_others), dim=1)