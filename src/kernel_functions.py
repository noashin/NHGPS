import jax.numpy as np
from .gps_jax import cov_map, exp_quadratic


def effects_kernel_with_decay(event_1, event_2, trial_ind_1, trial_ind_2,
                              observations_mat_1, observations_mat_2, hypers):
    """
    This function computes the entry of a kernel with decay over time.
    This function computes one entry in the kernel matrix. It should be used with JAX VMAP.
    :param event_1: scalar
    :param event_2: scalar
    :param trial_ind_1: trial index of event_1
    :param trial_ind_2: trial index of event_2
    :param observations_mat: matrix with all the observed events from all the trials padded with inf
    :param hypers: [kernel_output_variance, kernel_length_scale, memory_decay_factor]
    :return:  Entry in the kernel matrix
    """

    # TODO: add reference to equation on the paper

    kernel_output_variance, kernel_length_scale, memory_decay_factor = hypers
    diffs_vec_1 = event_1 - observations_mat_1[trial_ind_1]  # len(real_events)
    diffs_vec_2 = event_2 - observations_mat_2[trial_ind_2]  # len(real_events)

    # only consider the entries where event_1 > observed_event and event_2 > observed_event
    mask = (diffs_vec_1 > 0)[:, np.newaxis] * (diffs_vec_2 > 0)[np.newaxis, :]
    vec = kernel_output_variance * cov_map(exp_quadratic, diffs_vec_1 / (np.sqrt(2) * \
                                                                         kernel_length_scale),
                                           diffs_vec_2 / (np.sqrt(2) * kernel_length_scale)) * \
          np.exp(- memory_decay_factor * (diffs_vec_1[:, np.newaxis] + diffs_vec_2[np.newaxis, :]))
    zeros_vec = np.zeros(vec.shape, dtype=np.float64)
    vec = np.where(mask, vec, zeros_vec)

    return vec.sum()


def effects_kernel_with_decay_diag(event, trial_ind, observations_mat, hypers):
    """
    This function computes only the entries on the diagonal of the kernel.
    This function computes one entry in the diagonal vector. It should be used with JAX VMAP.
    This function is used only for the Variational Inference.
    :param event: scalar
    :param trial_ind: trial index of event_1
    :param observations_mat: matrix with all the observed events from all the trials padded with inf
    :param hypers: [kernel_output_variance, kernel_length_scale, memory_decay_factor]
    :return:  Entry in the kernel matrix
    """

    kernel_output_variance, kernel_length_scale, memory_decay_factor = hypers
    diffs_vec = event - observations_mat[trial_ind]  # len(real_events)

    mask = (diffs_vec > 0)[:, np.newaxis] * (diffs_vec > 0)[np.newaxis, :]
    vec = kernel_output_variance * cov_map(exp_quadratic, diffs_vec / (np.sqrt(2) * \
                                                                       kernel_length_scale),
                                           diffs_vec / (np.sqrt(2) * kernel_length_scale)) * \
          np.exp(- memory_decay_factor * (diffs_vec[:, np.newaxis] + diffs_vec[np.newaxis, :]))
    zeros_vec = np.zeros(vec.shape, dtype=np.float64)
    vec = np.where(mask, vec, zeros_vec)

    return vec.sum()


def grad_kernel_decay_wrt_memory_decay(event_1, event_2, trial_ind_1, trial_ind_2,
                                       observations_mat, hypers):
    """
    This function computes the gradient of the effects kernel with decay with respect to the kernel memory decay parameter.
    This function computes one entry in the gradient matrix. It should be used with JAX VMAP.
    :param event_1: scalar
    :param event_2: scalar
    :param trial_ind_1: trial index of event_1
    :param trial_ind_2: trial index of event_2
    :param observations_mat: matrix with all the observed events from all the trials padded with inf
    :param hypers: [kernel_output_variance, kernel_length_scale, memory_decay_factor]
    :return:  Gradient of an entry in the kernel matrix with respect to the length scale parameter.
    :return:
    """

    kernel_output_variance, kernel_length_scale, memory_decay_factor = hypers
    diffs_vec_1 = event_1 - observations_mat[trial_ind_1]  # len(real_events)
    diffs_vec_2 = event_2 - observations_mat[trial_ind_2]  # len(real_events)

    mask = (diffs_vec_1 > 0)[:, np.newaxis] * (diffs_vec_2 > 0)[np.newaxis, :]
    vec = kernel_output_variance * cov_map(exp_quadratic, diffs_vec_1 / (np.sqrt(2) * \
                                                                         kernel_length_scale),
                                           diffs_vec_2 / (np.sqrt(2) * kernel_length_scale)) * \
          np.exp(- memory_decay_factor * (diffs_vec_1[:, np.newaxis] + diffs_vec_2[np.newaxis, :])) * (
                  - diffs_vec_1[:, np.newaxis] - diffs_vec_2[np.newaxis, :])
    zeros_vec = np.zeros(vec.shape, dtype=np.float64)
    vec = np.where(mask, vec, zeros_vec)

    return vec.sum()


def grad_kernel_decay_wrt_kernel_ls(event_1, event_2, trial_ind_1, trial_ind_2,
                                    observations_mat, hypers):
    """
    This function computes the gradient of the effects kernel with decay with respect to the kernel length scale parameter.
    This function computes one entry in the gradient matrix. It should be used with JAX VMAP.
    :param event_1: scalar
    :param event_2: scalar
    :param trial_ind_1: trial index of event_1
    :param trial_ind_2: trial index of event_2
    :param observations_mat: matrix with all the observed events from all the trials padded with inf
    :param hypers: [kernel_length_scale, memory_decay_factor]
    :return:  Gradient of an entry in the kernel matrix with respect to the length scale parameter.
    """
    kernel_output_variance, kernel_length_scale, memory_decay_factor = hypers
    diffs_vec_1 = event_1 - observations_mat[trial_ind_1]
    diffs_vec_2 = event_2 - observations_mat[trial_ind_2]

    mask = (diffs_vec_1 > 0)[:, np.newaxis] * (diffs_vec_2 > 0)[np.newaxis, :]
    vec = kernel_output_variance * cov_map(exp_quadratic, diffs_vec_1 / (np.sqrt(2) * \
                                                                         kernel_length_scale),
                                           diffs_vec_2 / (np.sqrt(2) * kernel_length_scale)) * \
          np.exp(- memory_decay_factor * (diffs_vec_1[:, np.newaxis] + diffs_vec_2[np.newaxis, :])) * (
                  diffs_vec_1[:, np.newaxis] - diffs_vec_2[np.newaxis, :]) ** 2 / kernel_length_scale ** 3
    zeros_vec = np.zeros(vec.shape, dtype=np.float64)
    vec = np.where(mask, vec, zeros_vec)

    return vec.sum()

def grad_kernel_decay_wrt_kernel_out_var(event_1, event_2, trial_ind_1, trial_ind_2, observations_mat, hypers):
    """
    This function computes the gradient of the effects kernel with decay with respect to the kernel output variance parameter.
    This function computes one entry in the gradient matrix. It should be used with JAX VMAP.
    :param event_1: scalar
    :param event_2: scalar
    :param trial_ind_1: trial index of event_1
    :param trial_ind_2: trial index of event_2
    :param observations_mat: matrix with all the observed events from all the trials padded with inf
    :param hypers: [kernel_length_scale, memory_decay_factor]
    :return:  Gradient of an entry in the kernel matrix with respect to the output variance parameter.
    """
    kernel_amplitude, kernel_covariance, memory_decay = hypers
    diffs_vec_1 = event_1 - observations_mat[trial_ind_1]  # len(real_events)
    diffs_vec_2 = event_2 - observations_mat[trial_ind_2]  # len(real_events)

    mask = (diffs_vec_1 > 0)[:, np.newaxis] * (diffs_vec_2 > 0)[np.newaxis, :]
    vec = cov_map(exp_quadratic, diffs_vec_1 / (np.sqrt(2) * \
                                                kernel_covariance),
                  diffs_vec_2 / (np.sqrt(2) * kernel_covariance)) * \
          np.exp(- memory_decay * (diffs_vec_1[:, np.newaxis] + diffs_vec_2[np.newaxis, :]))
    zeros_vec = np.zeros(vec.shape, dtype=np.float64)
    vec = np.where(mask, vec, zeros_vec)

    return vec.sum()


def grad_kernel_decay_wrt_memory_decay_diag(event, trial_ind, observations_mat, hypers):
    """
    This function computes the diagonal vector of the gradient of the effects kernel with decay with respect
    to the kernel memory decay parameter.
    This function computes one entry in the gradient diagonal vector. It should be used with JAX VMAP.
    :param event: scalar
    :param trial_ind: trial index of event_1
    :param observations_mat: matrix with all the observed events from all the trials padded with inf
    :param kernel_output_variance: kernel output variance - usually set to 1
    :return:  Gradient of an entry in the kernel matrix with respect to the length scale parameter.
    :return:
    """

    kernel_output_variance, kernel_length_scale, memory_decay_factor = hypers
    diffs_vec = event - observations_mat[trial_ind]

    mask = (diffs_vec > 0)[:, np.newaxis] * (diffs_vec > 0)[np.newaxis, :]
    vec = kernel_output_variance * cov_map(exp_quadratic, diffs_vec / (np.sqrt(2) * \
                                                                       kernel_length_scale),
                                           diffs_vec / (np.sqrt(2) * kernel_length_scale)) * \
          np.exp(- memory_decay_factor * (diffs_vec[:, np.newaxis] + diffs_vec[np.newaxis, :])) * (
                  - diffs_vec[:, np.newaxis] - diffs_vec[np.newaxis, :])
    zeros_vec = np.zeros(vec.shape, dtype=np.float64)
    vec = np.where(mask, vec, zeros_vec)

    return vec.sum()


def grad_kernel_decay_wrt_kernel_ls_diag(event, trial_ind, observations_mat, hypers):
    """
    This function computes the diagonal vector of the gradient of the effects kernel with decay with respect
    to the kernel length scale parameter.
    This function computes one entry in the gradient diagonal vector. It should be used with JAX VMAP.
    :param event: scalar
    :param trial_ind: trial index of event_1
    :param observations_mat: matrix with all the observed events from all the trials padded with inf
    :param hypers: [kernel_output_variance, kernel_length_scale, memory_decay_factor]
    :return:  Gradient of an entry in the kernel matrix with respect to the length scale parameter.
    """

    kernel_output_variance, kernel_length_scale, memory_decay_factor = hypers
    diffs_vec = event - observations_mat[trial_ind]

    mask = (diffs_vec > 0)[:, np.newaxis] * (diffs_vec > 0)[np.newaxis, :]
    vec = kernel_output_variance * cov_map(exp_quadratic, diffs_vec / (np.sqrt(2) * kernel_length_scale),
                                           diffs_vec / (np.sqrt(2) * kernel_length_scale)) * \
          np.exp(- memory_decay_factor * (diffs_vec[:, np.newaxis] + diffs_vec[np.newaxis, :])) * (
                  diffs_vec[:, np.newaxis] - diffs_vec[np.newaxis, :]) ** 2 / kernel_length_scale ** 3
    zeros_vec = np.zeros(vec.shape, dtype=np.float64)
    vec = np.where(mask, vec, zeros_vec)

    return vec.sum()

def grad_kernel_decay_wrt_kernel_out_var_diag(event, trial_ind, observations_mat, hypers):
    """
    This function computes the diagonal vector of the gradient of the effects kernel with decay with respect
    to the kernel output variance parameter.
    This function computes one entry in the gradient diagonal vector. It should be used with JAX VMAP.
    :param event: scalar
    :param trial_ind: trial index of event_1
    :param observations_mat: matrix with all the observed events from all the trials padded with inf
    :param hypers: [kernel_output_variance, kernel_length_scale, memory_decay_factor]
    :return:  Gradient of an entry in the kernel matrix with respect to the output variance parameter.
    """
    kernel_amplitude, kernel_covariance, memory_decay = hypers
    diffs_vec_1 = event - observations_mat[trial_ind]  # len(real_events)

    mask = (diffs_vec_1 > 0)[:, np.newaxis] * (diffs_vec_1 > 0)[np.newaxis, :]
    vec = cov_map(exp_quadratic, diffs_vec_1 / (np.sqrt(2) * \
                                                                   kernel_covariance),
                                     diffs_vec_1 / (np.sqrt(2) * kernel_covariance)) * \
          np.exp(- memory_decay * (diffs_vec_1[:, np.newaxis] + diffs_vec_1[np.newaxis, :]))
    zeros_vec = np.zeros(vec.shape, dtype=np.float64)
    vec = np.where(mask, vec, zeros_vec)

    return vec.sum()


def rbf_kernel(event_1, event_2, hypers):
    """
    This function calculates an entry the RBF kernel of two events.
    :param event_1: scalar
    :param event_2: scalar
    :param hypers: [kernel_output_variance, kernel_length_scale]
    :return: scalar - entry in the RBF kernel.
    """

    # TODO: add reference to equation on the paper

    kernel_output_variance, kernel_length_scale = hypers
    return kernel_output_variance * exp_quadratic(event_1 / (np.sqrt(2) * \
                                                             kernel_length_scale),
                                                  event_2 / (np.sqrt(2) * kernel_length_scale))


def rbf_kernel_diag(events, hypers):
    """
    This function computes the diagonal vector of an RBF kernel.
    :param events: array of events
    :param hypesrs: [kernel_output_variance, kernel_length_scale]
    :return: scalar - entry in the RBF kernel diagonal.
    """
    kernel_output_variance, kernel_length_scale = hypers
    return kernel_output_variance * np.ones(events.shape[0], dtype=np.float64)


def kernel_with_decay_for_estimate_g(event_1, event_2, trial_ind_2, observations_mat, hypers):
    """
    This function computes the rbf kernel with decay between event_1 and event_2 - real_events.
    It is used in the estimation of the mean of \s and \g from the inferred intensity.
    :param event_1: scalar
    :param event_2: scalar
    :param trial_ind_2: trial index of event_2
    :param observations_mat: matrix with all the observed events from all the trials padded with inf
    :param hypers: [kernel_output_variance, kernel_length_scale, memory_decay_factor]
    :return: scalar - entry in the kernel matrix
    """

    # TODO: add reference to equation on the paper

    kernel_output_variance, kernel_length_scale, memory_decay = hypers
    diffs_vec_2 = event_2 - observations_mat[trial_ind_2]  # len(real_events)

    mask = diffs_vec_2 > 0
    vec = kernel_output_variance * np.exp(- (event_1 - diffs_vec_2) ** 2 / (np.sqrt(2) * kernel_length_scale)) * \
          np.exp(- memory_decay * (event_1 + diffs_vec_2))
    zeros_vec = np.zeros(vec.shape, dtype=np.float64)
    vec = np.where(mask, vec, zeros_vec)

    return vec.sum()


def kernel_with_decay_no_history_for_estimate_g(event_1, event_2, hypers):
    """
    This function computes the rbf kernel with decay between event_1 and event_2.
    It is used in the estimation of the covariance of \s and \g from the inferred intensity.
    :param event_1: scalar
    :param event_2: scalar
    :param hypers: [kernel_output_variance, kernel_length_scale, memory_decay_factor]
    :return: scalar - entry in the kernel matrix
    """

    # TODO: add reference to equation on the paper

    kernel_output_variance, kernel_length_scale, memory_decay = hypers

    return kernel_output_variance * exp_quadratic(event_1 / (np.sqrt(2) * \
                                                             kernel_length_scale),
                                                  event_1 / (np.sqrt(2) * kernel_length_scale)) * np.exp(
        - memory_decay * (event_1 + event_2))
