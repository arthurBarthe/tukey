import numpy as np
import torch
from scipy.stats import norm
import matplotlib.pyplot as plt
from tukey.losses import Tuckey_g_h_inverse
from tukey.utils import compute_pdf_tukey

# script parameters
LOG = False

# parameters of the transformation
g, h = -0.6, 0.3

# grid points
z_tildas = torch.linspace(-5, 5, 1000)
pdf = compute_pdf_tukey(z_tildas, g, h)

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(z_tildas, norm.pdf(z_tildas), linewidth=5, label='standard normal')
ax.plot(z_tildas, pdf, linewidth=5, linestyle='--', label='Tukey G-and-H')
if LOG:
    ax.set_yscale('log')
ax.set_xlabel('z')
ax.set_ylabel('probability density function')
ax.set_title(f'{g=}, {h=}')
ax.legend()
plt.show()