import itertools
# import ipdb

import numba
import jax

import numpy as np
import jax.numpy as jnp
from numba import jit as numba_jit
from jax import jit as jax_jit
from jax import vmap, config

def effects_kernel_with_decay(event_1, event_2, trial_ind_1, trial_ind_2,
                              observations_mat, kernel_output_variance, hypers):
    """
    This function computes the entry of a kernel with decay over time.
    This function computes one entry in the kernel matrix. It should be used with JAX VMAP.
    :param event_1: scalar
    :param event_2: scalar
    :param trial_ind_1: trial index of event_1
    :param trial_ind_2: trial index of event_2
    :param observations_mat: matrix with all the observed events from all the trials padded with inf
    :param kernel_output_variance: kernel output variance - usually set to 1
    :param hypers: [kernel_length_scale, memory_decay_factor]
    :return:  Entry in the kernel matrix
    """
    kernel_length_scale, memory_decay_factor = hypers
    diffs_vec_1 = event_1 - observations_mat[trial_ind_1]  # len(real_events)
    diffs_vec_2 = event_2 - observations_mat[trial_ind_2]  # len(real_events)

    # only consider the entries where event_1 > observed_event and event_2 > observed_event
    mask = (diffs_vec_1 > 0)[:, jnp.newaxis] * (diffs_vec_2 > 0)[jnp.newaxis, :]
    vec = kernel_output_variance * cov_map(exp_quadratic, diffs_vec_1 / (np.sqrt(2) * \
                                                                         kernel_length_scale),
                                           diffs_vec_2 / (jnp.sqrt(2) * kernel_length_scale)) * \
          jnp.exp(- memory_decay_factor * (diffs_vec_1[:, jnp.newaxis] + diffs_vec_2[jnp.newaxis, :]))
    zeros_vec = jnp.zeros(vec.shape, dtype=jnp.float64)
    vec = jnp.where(mask, vec, zeros_vec)

    return vec.sum()

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
    return jnp.exp(- jnp.sum((x1 - x2) ** 2))

@numba_jit(nopython=True)
def numba_func_jit(observations, events_1, events_2, hypers, num_events_1, num_events_2):
    kernel_covariance, memory_decay = hypers
    n1 = events_1.shape[0]
    n2 = events_2.shape[0]
    K = np.zeros((n1, n2))
    diffs1 = np.atleast_2d(events_1).T - np.atleast_2d(observations)
    diffs2 = np.atleast_2d(events_2).T - np.atleast_2d(observations)
    for i in range(n1):
        diffs_i = diffs1[i][num_events_1[0][i]:num_events_1[1][i]]
        diffs_i = diffs_i[diffs_i > 0]
        for j in range(n2):
            diffs_j = diffs2[j][num_events_2[0][j]:num_events_2[1][j]]
            diffs_j = diffs_j[diffs_j > 0]
            res = np.sum(np.exp(-(np.atleast_2d(diffs_i) - np.atleast_2d(diffs_j).T) ** 2 / (2. * kernel_covariance ** 2) \
                                                                                             - memory_decay * (np.atleast_2d(diffs_i) + np.atleast_2d(diffs_j).T)))
            K[i,j] = res
    return K

