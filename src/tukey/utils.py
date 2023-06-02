from scipy.stats import norm
import torch
from tukey.losses import Tuckey_g_h_inverse

def compute_pdf_tukey(z_tildas, g, h):
    tukey_inverse = Tuckey_g_h_inverse()
    g_ = torch.ones_like(z_tildas) * g
    h_ = torch.ones_like(z_tildas) * h
    zs = tukey_inverse.apply(z_tildas, g_, h_)
    f_zs = norm.pdf(zs)
    d_tau_d_zs = tukey_inverse.d_tau_d_z(zs, g_, h_)
    pdf = 1 / d_tau_d_zs * f_zs
    return pdf