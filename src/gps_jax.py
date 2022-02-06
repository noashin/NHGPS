import numpy as onp
from jax import vmap
import jax.numpy as np
import jax.random as random

# This code is taken from the here: https://github.com/google/jax/blob/master/examples/gaussian_process_regression.py

epsilon = 1e-5


def cov_map(cov_func, xs, xs2=None):
    """Compute a covariance matrix from a covariance function and data points.
    Args:
      cov_func: callable function, maps pairs of data points to scalars.
      xs: array of data points, stacked along the leading dimension.
    Returns:
      A 2d array `a` such that `a[i, j] = cov_func(xs[i], xs[j])`.
    """
    if xs2 is None:
        return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs)
    else:
        return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs2).T


def exp_quadratic(x1, x2):
    return np.exp(- np.sum((x1 - x2) ** 2))


def sample_gp(K, mean=0):
    """ Samples a GP.
    :param kernel: the GP's kernel
    :param mean: the GP's mean
    :return: numpy.ndarray [num_points]
        Function values of GP a the points.
    """
    num_points = K.shape[0]
    K += epsilon * np.eye(num_points)

    L = np.linalg.cholesky(K)

    seed = onp.random.randint(2 ** 32)
    key = random.PRNGKey(seed)

    rand_nums = random.normal(key, shape=(num_points,))
    gp = np.dot(L, rand_nums)
    gp += mean
    return gp

def rbf_kernel(x, x_prime, amplitude, length_scale):
    """ Computes the covariance functions between x and x_prime.
    :param x: num_points x D float32
        Contains coordinates for points of x
    :param x_prime: num_points_prime x D float32
        Contains coordinates for points of x_prime
    :param amplitude: scalar. amplitude of the kernel
    :param length_scale scalar.
    :return: num_points x num_points_prime
        Kernel matrix.
    """
    sqr_sum = (np.atleast_2d(x).T - np.atleast_2d(x_prime)) ** 2

    return amplitude * np.exp(- sqr_sum / (2. * length_scale ** 2))
