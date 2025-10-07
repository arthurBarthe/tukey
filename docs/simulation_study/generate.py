"""
In this file we define objects necessary to the generation of the simulated datasets used in our simulation study.

"""
import csv

import numpy as np
from numpy.random import default_rng, Generator
from scipy.stats import norm


class TukeyGHDistribution:
    def __init__(self):
        pass

    def x_to_theta(self, x):
        raise NotImplementedError()

    @staticmethod
    def tukey_g_h(z: float, theta: tuple):
        """Applies the tukey-g-h transform with parameters theta to z."""
        mu, beta, g, h = theta
        z_tilda = 1 / g * (np.exp(g * z) - 1) * np.exp(h * z ** 2 / 2)
        return mu + 1 / beta * z_tilda

    def sample(self, n_samples: int, rng: Generator = None):
        x_data = rng.uniform(0, 1, n_samples).astype(np.float32)
        z = rng.normal(0, 1, n_samples).astype(np.float32)
        theta = self.x_to_theta(x_data)
        y_data = TukeyGHDistribution.tukey_g_h(z, theta)
        return x_data, y_data

    def pdf(self, x_data: np.ndarray, y_data: np.ndarray):
        mu, beta, g, h = self.x_to_theta(x_data)
        tukey_inverse = Tuckey_g_h_inverse()
        z_tildas = (y_data - mu) * beta
        zs = tukey_inverse.apply(z_tildas, g, h)
        f_zs = norm.pdf(zs)
        d_tau_d_zs = tukey_inverse.d_tau_d_z(zs, g, h)
        pdf = 1 / d_tau_d_zs * f_zs * beta
        return pdf



class Distribution1(TukeyGHDistribution):
    def x_to_theta(self, x):
        return 20 * (x - 0.5) ** 2, 1 + 3 * (x - 0.25) ** 2, 1 / 2 * (x - 0.5), np.maximum(0, 0.25 - (x - 0.5) ** 2)


class SimulationDataset:
    def __init__(self, distribution, seed, sample_size, config_id: int = None, log=False):
        self.distribution = distribution
        self.seed = seed
        self.rng = default_rng(seed)
        self.sample_size = sample_size
        self.config_id = config_id
        if log:
            self.log()

    def sample(self):
        sample = self.distribution.sample(self.sample_size, rng=self.rng)
        self.rng = default_rng(self.seed)
        return sample

    def log(self):
        with open("simulation_study/datasets.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([self.distribution.__class__.__name__, self.seed, self.sample_size, self.config_id])

    @classmethod
    def load(cls, id: int):
        import sys
        with open("simulation_study/datasets.csv", mode="r", newline="") as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                if i == id:
                    return SimulationDataset(getattr(sys.modules[__name__], row[0])(), int(row[1]), int(row[2]))


if __name__ == '__main__':
    import secrets
    dist = Distribution1()
    for i in range(10):
        seed = secrets.randbits(128)
        dataset = SimulationDataset(dist, seed, 4000, 2, log=True)