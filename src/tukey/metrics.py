"""
In this file we define additional metrics used in the simulation study, in particular, the Kullback-Leibler divergence
"""

import numpy as np


class KLdivergence:
    """
    Class that implements the KL divergence.
    """
    def __init__(self, true_distribution):
        self.true_distribution = true_distribution

    def __call__(self, x_data, y_data, test_distribution):
        pdf_true = self.true_distribution.pdf(x_data, y_data)
        pdf_test = test_distribution.pdf(x_data, y_data)
        return np.mean((np.log(pdf_true) - np.log(pdf_test)) * pdf_true)

