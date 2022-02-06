import time
import pickle
import itertools

import numpy as onp
from scipy.integrate import quadrature
import jax.numpy as np
from jax import random, jit, vmap, device_put, value_and_grad, jacrev, random
from jax.scipy.special import digamma, gammaln, logsumexp
from jax.scipy.linalg import solve_triangular

from .kernel_functions import rbf_kernel, effects_kernel_with_decay, grad_kernel_decay_wrt_memory_decay, \
    grad_kernel_decay_wrt_kernel_ls, effects_kernel_with_decay_diag, rbf_kernel_diag, \
    grad_kernel_decay_wrt_kernel_ls_diag, grad_kernel_decay_wrt_memory_decay_diag, \
    grad_kernel_decay_wrt_kernel_out_var, grad_kernel_decay_wrt_kernel_out_var_diag, \
    kernel_with_decay_for_estimate_g, kernel_with_decay_no_history_for_estimate_g

from .helper_functions_for_variational_inference import calculate_postrior_GP, calculate_lower_bound, \
    partial_derivative, predictive_posterior_GP


class VI():
    def __init__(self, T, hyper_parameters, num_inducing_points, alpha_0=4, betta_0=0.06,
                 lmbda_star=None, conv_crit=1e-4,
                 num_integration_points=1000,
                 noise=1e-4):
        """

        :param T: temporal bound.
        :param hyper_parameters: [kernel_effects_output_variance, kernel_effects_length_sacle, kernel_effects_memory_decay,
                                  kernel_background_output_variance, kernel_background_length_scale]
        :param num_inducing_points: How many inducing points to use.
        :param alpha_0: shape value for the prior on the intensity bound.
        :param betta_0: 1. / scale value for the prior on the intensity bound.
        :param lmbda_star: ground truth value of the intensity value.
        :param conv_crit: convergence criteria
        :param num_integration_points: number of integration points
        :param noise: How much noise to add to the kernels.
        """

        self.T = T
        self.noise = noise
        self.hyper_params = hyper_parameters
        self.num_integration_points = num_integration_points
        self.num_inducing_points = num_inducing_points  # must be power of D

        self.alpha0 = alpha_0  # shape of the variational distribution of the intensity limit
        self.beta0 = betta_0  # 1 / scale of the variational distribution of the intensity limit
        if lmbda_star is None:
            self.lmbda_star_q1 = self.alpha0 / self.beta0
            self.log_lmbda_star_q1 = digamma(self.alpha0) - np.log(self.beta0)
        else:
            self.lmbda_star_q1 = lmbda_star
            self.log_lmbda_star_q1 = np.log(lmbda_star)
        self.alpha_q1 = self.alpha0
        self.beta_q1 = self.beta0
        self.convergence = np.inf  # ratio between the last two iterations
        self.conv_crit = conv_crit
        self.num_iterations = 0  # counter of the number of iterations

        # attributes related to the observations
        self.observations = None  # observations associated with the process - all trials 
        self.num_trials = 0  # numbe of trials
        self.observations_flat = None  # vector of all the observations concatenated.
        self.observations_padded_mat = None  # a matrix of all the observations padded with inf
        self.N_sum = 0  # number of observed observations in all trials
        self.observations_trial_inds = None  # vector of theindices of the trial of each observation in self.observations_flat

        self.LB_list = []  # list of the values of the lower bound in each iteration
        self.times = []  # list of the time of each iteration
        self.hyper_params_list = []  # list of the values of the hyper parameters in each iteration

        self.grads = 0
        # attributes for the ADAM optimizer
        self.epsilon_grad = 1e-8
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.m = 0.
        self.v = 0.
        self.t = 1.

        # attributes of the variational inference
        self.induced_points = None  # list with induced points for each trial.
        self.induced_points_flat = None  # Flat vector with induced points for each trial.
        self.integration_points = None  # list of integration points for each trial.
        self.induced_points_trials_inds = None  # trial index for each induced point.
        self.integration_points_flat = None  # Flat vector with the integration points for each trial.
        self.integration_points_trials_inds = None  # trial index for each integration point.
        self.num_integration_points_sum = 0  # Total number of integration points for all trials.

        self.mu_omega_X = None  # mean of the variational distribution over the augmenting PG variables with respect to the observations.
        self.mu_omega_int_points = None  # mean of the variational distribution over the augmenting PG variables with respect to the integration points.
        self.lmbda_q2 = None  # The distribution over the augmenting events.

        self.mu_g_s = None  # mean of the variational distribution in the inducing points
        self.Sigma_g_s = None  # covariance of the variational distribution in the inducing points.
        self.Sigma_g_s_inv = None  # inverse of the covariance of the variational distribution in the inducing points.
        self.Kss_background = None  # The kernel of the background rate for the inducing points
        self.Kss = None  # kernel of the inducing points
        self.Kss_inv = None  # inverse of the kernel of the inducing points.
        self.logdet_Ks = None  # log of the determinant of the kernel of the inducing points

        self.Kxx_background_diag = None  # the diagonal of the background rate kernel of the observations
        self.Kxx_diag = None  # the diagonal of the kernel of the observations
        self.Ksx_background = None  # background rate kernel of the inducing points and the observations.
        self.ks_X = None  # kernel of the inducing points and the observations.
        self.kappa_X = None  # Kss_inv * ks_X
        self.mu_g_X = None  # mean of the variational distribution over the observations
        self.var_g_X = None  # variance of the variational distribution over the observations
        self.mu_g2_X = None  # second moment of the variational distribution over the observations.

        self.Kii_background_diag = None  # The diagonal of the background rate kernel of the integration points
        self.Kii_diag = None  # background rate kernel of the integration points
        self.Ksi_background = None  # background rate kernel of the inducing points and the integration points.
        self.ks_int_points = None  # kernel of the inducing points and the integration points.
        self.kappa_int_points = None  # Kss_inv * ks_int_points
        self.mu_g_int_points = None  # mean of the variational distribution over the integration points
        self.var_g_int_points = None  # variance of the variational distribution over the integration points
        self.mu_g2_int_points = None  # second moment of the variational distribution over the integration points

        self.jit_phi_kernel_background = jit(vmap(vmap(rbf_kernel, (0, None, None), 0), (None, 0, None), 0))
        self.jit_phi_kernel_background_diag = jit(rbf_kernel_diag)
        self.jit_phi_kernel_effects = jit(
            vmap(vmap(effects_kernel_with_decay, (0, None, 0, None, None, None, None), 0),
                 (None, 0, None, 0, None, None, None),
                 0))
        self.jit_phi_kernel_effects_diag = jit(vmap(effects_kernel_with_decay_diag, (0, 0, None, None), 0))
        self.jit_kernel_with_decay_for_estimate_g = jit(
            vmap(vmap(kernel_with_decay_for_estimate_g, (0, None, None, None, None), 0),
                 (None, 0, 0, None, None),
                 0))
        self.jit_kernel_with_decay_no_history_for_estimate_g = jit(
            vmap(vmap(kernel_with_decay_no_history_for_estimate_g, (0, None, None), 0),
                 (None, 0, None),
                 0))

        self.jit_grad_kernel_decay_wrt_memory_decay = jit(
            vmap(vmap(grad_kernel_decay_wrt_memory_decay, (0, None, 0, None, None, None), 0),
                 (None, 0, None, 0, None, None),
                 0))
        self.jit_grad_kernel_decay_wrt_kernel_ls = jit(
            vmap(vmap(grad_kernel_decay_wrt_kernel_ls, (0, None, 0, None, None, None), 0),
                 (None, 0, None, 0, None, None),
                 0))
        self.jit_grad_kernel_decay_wrt_kernel_out_var = jit(
            vmap(vmap(grad_kernel_decay_wrt_kernel_out_var, (0, None, 0, None, None, None), 0),
                 (None, 0, None, 0, None, None),
                 0))

        self.jit_grad_kernel_decay_wrt_kernel_ls_diag = jit(
            vmap(grad_kernel_decay_wrt_kernel_ls_diag, (0, 0, None, None), 0))
        self.jit_grad_kernel_decay_wrt_memory_decay_diag = jit(
            vmap(grad_kernel_decay_wrt_memory_decay_diag, (0, 0, None, None), 0))
        self.jit_grad_kernel_decay_wrt_kernel_out_var_diag = jit(
            vmap(grad_kernel_decay_wrt_kernel_out_var_diag, (0, 0, None, None), 0))

        self.jit_partial_derivative = jit(vmap(partial_derivative, (2, 2, 2, 1, 1, None, None, None, None, None, None,
                                                                    None, None, None, None, None, None), 0))
        self.jit_calculate_postrior_GP = jit(calculate_postrior_GP)
        self.jit_predictive_posterior_GP = jit(predictive_posterior_GP)
        self.jit_calculate_lower_bound = jit(calculate_lower_bound)

        self.seed = onp.random.randint(0, 2 ** 18)
        self.key = random.PRNGKey(self.seed)

    def set_data(self, observations):
        '''
        This methods set the observations attribut of the model
        :param observations: shape - num_trials x num_events (list of lists of lists)
        :return:
        '''
        self.observations = observations
        self.num_trials = len(self.observations)
        self.observations_flat = np.hstack(self.observations)
        try:
            self.observations_padded_mat = np.array(
                list(itertools.zip_longest(*self.observations, fillvalue=np.inf))).T
        except:
            self.observations_padded_mat = np.atleast_2d(self.observations)

        self.N = np.array([len(self.observations[i]) for i in range(self.num_trials)])
        self.N_sum = np.sum(self.N)
        self.observations_trial_inds = np.hstack(
            [np.repeat(n, self.observations[n].shape[0]) for n in range(self.num_trials)])

    def initialize_for_inference(self):
        """
        This methods initializes the values for the variational distribution parameters.
        :return:
        """
        print('placing inducing points')
        self.place_inducing_points()
        print('placing integration points')
        self.place_integration_points()

        print('init kernels')
        self.update_kernels()

        print('init mus and sigs')
        self.mu_g_s = np.zeros(self.induced_points_flat.shape[0], dtype=np.float64)
        self.Sigma_g_s = np.identity(self.induced_points_flat.shape[0], dtype=np.float64)
        self.Sigma_g_s_inv = np.identity(self.induced_points_flat.shape[0], dtype=np.float64)

        self.mu_g_X, var_g_X = self.jit_predictive_posterior_GP(self.ks_X, self.kappa_X, self.mu_g_s, self.Sigma_g_s,
                                                                self.Kxx_diag)

        self.mu_g2_X = var_g_X + self.mu_g_X ** 2
        self.mu_g_int_points, var_g_int_points = self.jit_predictive_posterior_GP(self.ks_int_points,
                                                                                  self.kappa_int_points, self.mu_g_s,
                                                                                  self.Sigma_g_s, self.Kii_diag)
        self.mu_g2_int_points = var_g_int_points + self.mu_g_int_points ** 2

    def place_inducing_points(self):
        """"
        This method places self.num_induced_points on a grid for each trial.
        """
        dist_between_points = self.T / self.num_inducing_points
        induced_points_one_trial = np.arange(.5 * dist_between_points, self.T, dist_between_points,
                                             dtype=np.float64)
        self.induced_points = np.repeat(induced_points_one_trial[np.newaxis, :], self.num_trials, axis=0)
        self.induced_points_flat = np.hstack(self.induced_points)
        self.induced_points_trials_inds = np.hstack(
            [np.repeat(n, self.induced_points[n].shape[0]) for n in range(self.num_trials)])

    def place_integration_points(self):
        """ This method places the integration points for the Monte Carlo integration for each trial.
        The points are sampled uniformly.
        """

        integration_points_one_trial = random.uniform(self.key, (self.num_integration_points,)) * self.T
        self.integration_points = np.repeat(integration_points_one_trial[np.newaxis, :], self.num_trials, axis=0)
        self.integration_points_flat = np.hstack(self.integration_points)
        self.integration_points_trials_inds = np.hstack(
            [np.repeat(n, self.integration_points[n].shape[0]) for n in range(self.num_trials)])

        self.num_integration_points_sum = self.num_integration_points * self.num_trials

    def update_kernels(self):
        """
        This method updates the kernels following an update of the hyper parameters.
        """
        Kss_effects = self.jit_phi_kernel_effects(self.induced_points_flat, self.induced_points_flat,
                                                  self.induced_points_trials_inds, self.induced_points_trials_inds,
                                                  self.observations_padded_mat, self.observations_padded_mat,
                                                  self.hyper_params[:3])
        self.Kss_background = self.jit_phi_kernel_background(self.induced_points_flat, self.induced_points_flat,
                                                             self.hyper_params[3:])
        self.Kss = self.Kss_background + Kss_effects
        L = np.linalg.cholesky(self.Kss + self.noise * np.eye(self.Kss.shape[0], dtype=np.float64))
        L_inv = solve_triangular(L, np.eye(L.shape[0], dtype=np.float64), lower=True, check_finite=False)
        self.Kss_inv = L_inv.T.dot(L_inv)
        self.logdet_Ks = 2. * np.sum(np.log(L.diagonal()))

        Kxx_effects_diag = self.jit_phi_kernel_effects_diag(self.observations_flat, self.observations_trial_inds,
                                                            self.observations_padded_mat,
                                                            self.hyper_params[:3])
        self.Kxx_background_diag = self.jit_phi_kernel_background_diag(self.observations_flat, self.hyper_params[3:])
        self.Kxx_diag = Kxx_effects_diag + self.Kxx_background_diag

        Kii_effects_diag = self.jit_phi_kernel_effects_diag(self.integration_points_flat,
                                                            self.integration_points_trials_inds,
                                                            self.observations_padded_mat,
                                                            self.hyper_params[:3])
        self.Kii_background_diag = self.jit_phi_kernel_background_diag(self.integration_points_flat,
                                                                       self.hyper_params[3:])
        self.Kii_diag = Kii_effects_diag + self.Kii_background_diag

        Ksi_effects = self.jit_phi_kernel_effects(self.integration_points_flat, self.induced_points_flat,
                                                  self.integration_points_trials_inds,
                                                  self.induced_points_trials_inds,
                                                  self.observations_padded_mat, self.observations_padded_mat,
                                                  self.hyper_params[:3])
        self.Ksi_background = self.jit_phi_kernel_background(self.integration_points_flat, self.induced_points_flat,
                                                             self.hyper_params[3:])
        self.ks_int_points = self.Ksi_background + Ksi_effects

        Ksx_effects = self.jit_phi_kernel_effects(self.observations_flat, self.induced_points_flat,
                                                  self.observations_trial_inds, self.induced_points_trials_inds,
                                                  self.observations_padded_mat, self.observations_padded_mat,
                                                  self.hyper_params[:3])
        self.Ksx_background = self.jit_phi_kernel_background(self.observations_flat, self.induced_points_flat,
                                                             self.hyper_params[3:])
        self.ks_X = self.Ksx_background + Ksx_effects

        self.kappa_int_points = self.Kss_inv.dot(self.ks_int_points)
        self.kappa_X = self.Kss_inv.dot(self.ks_X)

    def update_hypers(self, step):
        """
        This method updates the hyper parameters according to the ADAM step.
        :param step: step for the hyper parameters
        :return:
        """
        self.hyper_params = self.hyper_params - step

    def adam_grad_step(self):
        """
        This method computes the gradient step following the ADAM optimizer.
        :return The gradient step
        """
        self.m = self.beta_1 * self.m + (1. - self.beta_1) * self.grads
        self.v = self.beta_2 * self.v + (1. - self.beta_2) * np.power(self.grads, 2)
        m_hat = self.m / (1 - np.power(self.beta_1, self.t))
        v_hat = self.v / (1 - np.power(self.beta_2, self.t))
        return m_hat / (np.sqrt(v_hat) + self.epsilon_grad)

    def gradient_step(self, grad_step_size, adapt_grad_step_size):
        """
        This method clculates the grad step based on the ADAM step and the step size.
        :param grad_step_size: step size of the gradient
        :param adapt_grad_step_size: whether to adapt the step size or not
        """
        if adapt_grad_step_size and self.t > 0:
            eff_grad_step_size = grad_step_size / np.sqrt(self.t)
        else:
            eff_grad_step_size = grad_step_size
        adam_step = self.adam_grad_step()
        step = eff_grad_step_size * adam_step
        self.update_hypers(step)

    def calc_gradients(self):
        """
        This method calculates the gradient of the ELBO with respect to the model hyper parameters
        and updates self.grads accordingly.
        """

        kernel_effects_output_variance, kernel_effects_length_sacle, kernel_effects_memory_decay, memory_decay, \
        kernel_background_output_variance, kernel_background_length_scale = self.hyper_params
        Sigma_s_mugmug = self.Sigma_g_s + np.outer(self.mu_g_s, self.mu_g_s)

        # derivatives of effects kernel params
        # kernel derivatives wrt kernel_effects_output_variance
        dKxx_wrt_ke_amp_diag = self.jit_grad_kernel_decay_wrt_kernel_out_var_diag(self.observations_flat,
                                                                                  self.observations_trial_inds,
                                                                                  self.observations_padded_mat,
                                                                                  self.hyper_params[:3])
        dKii_wrt_ke_amp_diag = self.jit_grad_kernel_decay_wrt_kernel_out_var_diag(self.integration_points_flat,
                                                                                  self.integration_points_trials_inds,
                                                                                  self.observations_padded_mat,
                                                                                  self.hyper_params[:3])

        dks_X_wrt_ke_amp = self.jit_grad_kernel_decay_wrt_kernel_out_var(self.observations_flat,
                                                                         self.induced_points_flat,
                                                                         self.observations_trial_inds,
                                                                         self.induced_points_trials_inds,
                                                                         self.observations_padded_mat,
                                                                         self.hyper_params[:3])
        dks_int_points_wrt_ke_amp = self.jit_grad_kernel_decay_wrt_kernel_out_var(self.integration_points_flat,
                                                                                  self.induced_points_flat,
                                                                                  self.integration_points_trials_inds,
                                                                                  self.induced_points_trials_inds,
                                                                                  self.observations_padded_mat,
                                                                                  self.hyper_params[:3])
        dKs_wrt_ke_amp = self.jit_grad_kernel_decay_wrt_kernel_out_var(self.induced_points_flat,
                                                                       self.induced_points_flat,
                                                                       self.induced_points_trials_inds,
                                                                       self.induced_points_trials_inds,
                                                                       self.observations_padded_mat,
                                                                       self.hyper_params[:3])

        # kernel derivatives wrt kernel_lengthscale
        dKxx_wrt_ke_ls_diag = self.jit_grad_kernel_decay_wrt_kernel_ls_diag(self.observations_flat,
                                                                            self.observations_trial_inds,
                                                                            self.observations_padded_mat,
                                                                            self.hyper_params[:3])
        dKii_wrt_ke_ls_diag = self.jit_grad_kernel_decay_wrt_kernel_ls_diag(self.integration_points_flat,
                                                                            self.integration_points_trials_inds,
                                                                            self.observations_padded_mat,
                                                                            self.hyper_params[:3])

        dks_X_wrt_ke_ls = self.jit_grad_kernel_decay_wrt_kernel_ls(self.observations_flat, self.induced_points_flat,
                                                                   self.observations_trial_inds,
                                                                   self.induced_points_trials_inds,
                                                                   self.observations_padded_mat,
                                                                   self.hyper_params[:3])
        dks_int_points_wrt_ke_ls = self.jit_grad_kernel_decay_wrt_kernel_ls(self.integration_points_flat,
                                                                            self.induced_points_flat,
                                                                            self.integration_points_trials_inds,
                                                                            self.induced_points_trials_inds,
                                                                            self.observations_padded_mat,
                                                                            self.hyper_params[:3])
        dKs_wrt_ke_ls = self.jit_grad_kernel_decay_wrt_kernel_ls(self.induced_points_flat, self.induced_points_flat,
                                                                 self.induced_points_trials_inds,
                                                                 self.induced_points_trials_inds,
                                                                 self.observations_padded_mat, self.hyper_params[:3])

        # kernel derivatives wrt memory decay factor
        dKxx_wrt_ke_md_diag = self.jit_grad_kernel_decay_wrt_memory_decay_diag(self.observations_flat,
                                                                               self.observations_trial_inds,
                                                                               self.observations_padded_mat,
                                                                               self.hyper_params[:3])
        dKii_wrt_ke_md_diag = self.jit_grad_kernel_decay_wrt_memory_decay_diag(self.integration_points_flat,
                                                                               self.integration_points_trials_inds,
                                                                               self.observations_padded_mat,
                                                                               self.hyper_params[:3])
        dks_X_wrt_ke_md = self.jit_grad_kernel_decay_wrt_memory_decay(self.observations_flat, self.induced_points_flat,
                                                                      self.observations_trial_inds,
                                                                      self.induced_points_trials_inds,
                                                                      self.observations_padded_mat,
                                                                      self.hyper_params[:3])
        dks_int_points_wrt_ke_md = self.jit_grad_kernel_decay_wrt_memory_decay(self.integration_points_flat,
                                                                               self.induced_points_flat,
                                                                               self.integration_points_trials_inds,
                                                                               self.induced_points_trials_inds,
                                                                               self.observations_padded_mat,
                                                                               self.hyper_params[:3])
        dKs_wrt_ke_md = self.jit_grad_kernel_decay_wrt_memory_decay(self.induced_points_flat, self.induced_points_flat,
                                                                    self.induced_points_trials_inds,
                                                                    self.induced_points_trials_inds,
                                                                    self.observations_padded_mat,
                                                                    self.hyper_params[:3])

        # derivatives of background kernel params
        # kernel derivatives wrt kernel s output variance
        dKxx_wrt_kb_a = self.Kxx_background_diag / kernel_background_output_variance
        dKii_wrt_kb_a = self.Kii_background_diag / kernel_background_output_variance
        dks_X_wrt_kb_a = self.Ksx_background / kernel_background_output_variance
        dks_int_points_wrt_kb_a = self.Ksi_background / kernel_background_output_variance
        dKs_wrt_kb_a = self.Kss_background / kernel_background_output_variance

        # kernel derivatives wrt kernel s length scale
        dKxx_wrt_kb_ls_diag = np.zeros(self.observations_flat.shape[0], dtype=np.float64)
        dKii_wrt_kb_ls_diag = np.zeros(self.integration_points_flat.shape[0], dtype=np.float64)
        dx = np.subtract(self.induced_points_flat[:, None], self.observations_flat[None])
        dks_X_wrt_kb_ls = self.Ksx_background[:, :] * (dx ** 2) / (kernel_background_length_scale ** 3)
        dx = np.subtract(self.induced_points_flat[:, None], self.integration_points_flat[None])
        dks_int_points_wrt_kb_ls = self.Ksi_background[:, :] * (dx ** 2) / (kernel_background_length_scale ** 3)
        dx = np.subtract(self.induced_points_flat[:, None], self.induced_points_flat[None])
        dKs_wrt_kb_ls = self.Kss_background[:, :] * (dx ** 2) / (kernel_background_length_scale ** 3)

        dks_X = np.stack([dks_X_wrt_ke_amp, dks_X_wrt_ke_ls, dks_X_wrt_ke_md, dks_X_wrt_kb_a, dks_X_wrt_kb_ls], axis=-1)
        dks_int_points = np.stack(
            [dks_int_points_wrt_ke_amp, dks_int_points_wrt_ke_ls, dks_int_points_wrt_ke_md, dks_int_points_wrt_kb_a,
             dks_int_points_wrt_kb_ls],
            axis=-1)
        dKs = np.stack([dKs_wrt_ke_amp, dKs_wrt_ke_ls, dKs_wrt_ke_md, dKs_wrt_kb_a, dKs_wrt_kb_ls], axis=-1)
        dKxx_diag = np.stack(
            [dKxx_wrt_ke_amp_diag, dKxx_wrt_ke_ls_diag, dKxx_wrt_ke_md_diag, dKxx_wrt_kb_a, dKxx_wrt_kb_ls_diag],
            axis=-1)
        dKii_diag = np.stack(
            [dKii_wrt_ke_amp_diag, dKii_wrt_ke_ls_diag, dKii_wrt_ke_md_diag, dKii_wrt_kb_a, dKii_wrt_kb_ls_diag],
            axis=-1)

        dL_dtheta = self.jit_partial_derivative(
            dKs, dks_X, dks_int_points, dKxx_diag, dKii_diag, self.Kss_inv, self.ks_int_points, self.kappa_X, self.ks_X,
            self.kappa_int_points, self.mu_g_s, Sigma_s_mugmug, self.mu_omega_X, self.mu_omega_int_points,
            self.lmbda_q2, self.num_integration_points_sum, self.T)

        self.grads = - np.array(dL_dtheta)

    def run(self, save_steps=False, file_path='', hyper_parms_inference=False, infer_max_intensity=True,
            grad_step_size=0.01, adapt_grad_step_size=False, hyper_updates=1, min_num_iterations=1,
            output=True):
        """
        This method performs the variational inference.
        :param save_steps: Whether to save intermediate states, boolean
        :param file_path: Where to save the results, string.
        :param hyper_parms_inference: Whether to infer the hyper parameters or not. boolean.
        :param infer_max_intensity:  Whether to infer the intensity limit.
        :param grad_step_size: Step size for the gradient.
        :param adapt_grad_step_size: Whether to adapt the gradient step size.
        :param hyper_updates: How many gradient step to take with respect to the hyper parameters at each iteration.
        :param min_num_iterations: Minimum number of VI iterations.
        :param output: Whether to display output during the run or not.
        """
        assert self.observations is not None, "Please set the data"

        # Initialisation
        self.times.append(time.perf_counter())
        self.initialize_for_inference()
        self.calculate_PG_expectations()
        self.calculate_posterior_intensity()

        converged = False

        self.hyper_params_list.append(self.hyper_params)

        while not converged or self.num_iterations < min_num_iterations:
            self.num_iterations += 1
            # Update q2
            self.Sigma_g_s_inv, self.Sigma_g_s, self.logdet_Sigma_g_s, self.mu_g_s = self.jit_calculate_postrior_GP(
                self.lmbda_q2, self.mu_omega_int_points, self.kappa_X, self.mu_omega_X,
                self.kappa_int_points, self.num_integration_points_sum, self.T, self.Kss_inv, self.noise,
                self.ks_X, self.ks_int_points)

            self.update_predictive_posterior()
            if infer_max_intensity:
                self.update_max_intensity()

            # Update q1
            self.calculate_PG_expectations()
            self.calculate_posterior_intensity()

            # update the hyper parameters
            if hyper_parms_inference:
                for i in range(hyper_updates):
                    self.calc_gradients()
                    self.gradient_step(grad_step_size, adapt_grad_step_size)
                    self.update_kernels()
                    self.update_predictive_posterior()
                    self.hyper_params_list.append(self.hyper_params)

            # Calculate the lower bound
            lb = self.jit_calculate_lower_bound(self.Sigma_g_s, self.mu_g_s, self.mu_g_int_points,
                                                self.mu_g2_int_points, self.mu_omega_int_points,
                                                self.lmbda_q2, self.c_int_points, self.log_lmbda_star_q1, self.mu_g_X,
                                                self.mu_g2_X, self.mu_omega_X, self.c_X,
                                                self.num_integration_points_sum, self.T, self.lmbda_star_q1,
                                                self.Kss_inv,
                                                self.logdet_Ks, self.logdet_Sigma_g_s,
                                                self.num_inducing_points, self.alpha0, self.beta0, self.alpha_q1,
                                                self.beta_q1)
            self.LB_list.append(lb)

            # Check for convergence
            if self.num_iterations > 1:
                self.convergence = np.absolute(self.LB_list[-1] -
                                               self.LB_list[
                                                   -2]) / max([np.abs(self.LB_list[-1]),
                                                               np.abs(self.LB_list[-2]), 1])
            converged = self.conv_crit > self.convergence
            self.times.append(time.perf_counter())
            if output and not self.num_iterations % 10:
                self.print_info()

            if save_steps and not self.num_iterations % 50:
                with open(file_path, 'wb') as f:
                    pickle.dump([self.LB_list, self.mu_g_X, self.mu_g2_X, self.hyper_params_list,
                                 self.induced_points, self.integration_points, self.Kss_inv, self.ks_int_points,
                                 self.ks_X, self.observations, self.Sigma_g_s, self.mu_g_s, self.lmbda_star_q1,
                                 self.alpha_q1,
                                 self.beta_q1], f)

        if save_steps:
            with open(file_path, 'wb') as f:
                pickle.dump([self.LB_list, self.mu_g_X, self.mu_g2_X, self.hyper_params_list,
                             self.induced_points, self.integration_points, self.Kss_inv, self.ks_int_points,
                             self.ks_X, self.observations, self.Sigma_g_s, self.mu_g_s, self.lmbda_star_q1,
                             self.alpha_q1,
                             self.beta_q1], f)

    def print_info(self):
        """
        This method prints the current state -
        how many iterations where done and what's the current convergence.
        """
        print((' +-----------------+ ' +
               '\n |  Iteration %4d |' +
               '\n |  Conv. = %.5f |' +
               '\n +-----------------+') % (self.num_iterations,
                                            self.convergence))

    def calculate_PG_expectations(self):
        """ This method updates the Polya-Gamma variational posterior is updated.
        """

        self.c_X = np.sqrt(self.mu_g2_X)
        self.mu_omega_X = .5 / self.c_X * np.tanh(.5 * self.c_X)
        self.c_int_points = np.sqrt(self.mu_g2_int_points)
        self.mu_omega_int_points = .5 / self.c_int_points * np.tanh(.5 * self.c_int_points)

    def calculate_posterior_intensity(self):
        """ This method updates the rate of the posterior process.
        """

        self.lmbda_q2 = .5 * np.exp(
            -.5 * self.mu_g_int_points + self.log_lmbda_star_q1) / \
                        np.cosh(.5 * self.c_int_points)

    def update_predictive_posterior(self, only_int_points=False):
        """
        This method Updates the mean and variance of the variational distribution over \phi at each point
        (observed and points for monte carlo integral)
        :param only_int_points: bool
            If True it only updates the integration points. (Default=False)
        """

        if not only_int_points:
            mu_g_X, var_g_X = self.jit_predictive_posterior_GP(self.ks_X, self.kappa_X, self.mu_g_s, self.Sigma_g_s,
                                                               self.Kxx_diag)
            self.mu_g_X = mu_g_X
            self.mu_g2_X = var_g_X + mu_g_X ** 2
        mu_g_int_points, var_g_int_points = self.jit_predictive_posterior_GP(self.ks_int_points, self.kappa_int_points,
                                                                             self.mu_g_s, self.Sigma_g_s, self.Kii_diag)
        self.mu_g_int_points = mu_g_int_points
        self.mu_g2_int_points = var_g_int_points + mu_g_int_points ** 2

    def update_max_intensity(self):
        """
        This method updates the variational distribution  for the maximal intensity.
        """
        self.alpha_q1 = self.N_sum + np.sum(
            self.lmbda_q2) / self.num_integration_points_sum * self.T + self.alpha0
        self.beta_q1 = self.beta0 + self.T
        self.lmbda_star_q1 = self.alpha_q1 / self.beta_q1
        self.log_lmbda_star_q1 = digamma(self.alpha_q1) - \
                                 np.log(self.beta_q1)

    def prepare_for_pred_dens(self, X_eval_flat, X_eval_trial_inds):
        """
        This method prepares the kernels necessary for the predictive intensity.
        :param X_eval_flat: flat vector with a grid for each trial.
        :param X_eval_trial_inds: indices of the trials for each point in he grid.
        :return: kernel of the inducing points ans the grid, the diagonal of the kernel of the grid
        """

        # kernel of the inducing points and the grid
        ks_g_effects = self.jit_phi_kernel_effects(X_eval_flat, self.induced_points_flat,
                                                   X_eval_trial_inds, self.induced_points_trials_inds,
                                                   self.observations_padded_mat, self.observations_padded_mat,
                                                   self.hyper_params[:3])
        ks_g_background = self.jit_phi_kernel_background(X_eval_flat, self.induced_points_flat,
                                                         self.hyper_params[3:])
        ks_g = ks_g_background + ks_g_effects

        # diagonal of the kernel of the grid
        kgg_effects_diag = self.jit_phi_kernel_effects_diag(X_eval_flat, X_eval_trial_inds,
                                                            self.observations_padded_mat,
                                                            self.hyper_params[:3])
        kgg_background_diag = self.jit_phi_kernel_background_diag(X_eval_flat, self.hyper_params[3:])
        k_gg_diag = kgg_background_diag + kgg_effects_diag

        return ks_g, k_gg_diag

    def predictive_intensity_function(self, X_eval):
        """
        This method estimates the intensity function on some points.
        :param X_eval: Grid on which to evaluate the intensity funciton.
        X_eval should be a list of length num_trials with each entry a grid between o and time_bound.
        :return: mean and variance of the intensity estimates in X_eval
        """

        X_eval_trial_inds = np.hstack([np.repeat(n, X_eval[n].shape[0]) for n in range(self.num_trials)])
        X_eval_flat = np.hstack(X_eval)
        num_preds = X_eval_flat.shape[0]

        # prepare the kernels
        self.kappa_int_points = self.Kss_inv.dot(self.ks_int_points)
        self.kappa_X = self.Kss_inv.dot(self.ks_X)
        ks_g, k_gg_diag = self.prepare_for_pred_dens(X_eval_flat, X_eval_trial_inds)
        kappa = self.Kss_inv.dot(ks_g)
        mu_pred, var_pred = self.jit_predictive_posterior_GP(ks_g, kappa, self.mu_g_s, self.Sigma_g_s,
                                                             k_gg_diag)

        # initialize the mean and the variance
        mean_lmbda_pred, var_lmbda_pred = onp.empty(num_preds, dtype=np.float64), onp.empty(num_preds, dtype=np.float64)

        # intensity limit
        mean_lmbda_q1 = self.lmbda_star_q1 / self.num_trials
        var_lmbda_q1 = self.alpha_q1 / (self.beta_q1 ** 2)
        var_lmbda_q1 /= self.num_trials ** 2
        mean_lmbda_q1_squared = var_lmbda_q1 + mean_lmbda_q1 ** 2

        for ipred in range(num_preds):
            print(ipred)
            mu, std = mu_pred[ipred], np.sqrt(var_pred[ipred])
            func1 = lambda g_pred: 1. / (1. + onp.exp(-g_pred)) * \
                                   onp.exp(-.5 * (g_pred - mu) ** 2 / std ** 2) / \
                                   onp.sqrt(2. * np.pi * std ** 2)
            a, b = mu - 10. * std, mu + 10. * std
            mean_lmbda_pred[ipred] = mean_lmbda_q1 * quadrature(func1, a, b,
                                                                maxiter=500)[0]
            func2 = lambda g_pred: (1. / (1. + np.exp(- g_pred))) ** 2 * \
                                   onp.exp(
                                       -.5 * (g_pred - mu) ** 2 / std ** 2) / \
                                   onp.sqrt(2. * onp.pi * std ** 2)
            a, b = mu - 10. * std, mu + 10. * std
            mean_lmbda_pred_squared = mean_lmbda_q1_squared * \
                                      quadrature(func2, a, b, maxiter=500)[0]
            var_lmbda_pred[ipred] = mean_lmbda_pred_squared - mean_lmbda_pred[
                ipred] ** 2

        return mean_lmbda_pred, var_lmbda_pred

    def prepare_test_data_for_ll(self, test_set, T=0):
        """
        This method prepares the data structures necessary for the log likelihood calculation
        :param test_set: list of trials of events in the test set.
        :param T: time limit for the events in the test set.
        :return: integration points for the test set, flattened test set, trial indices for the integration points,
                 trial indices for the test set, the test data in a padded matrix form, number of trials in the test set,
                 time bound of the test set
        """
        test_set_num_trials = len(test_set)
        test_set_flat = np.hstack(test_set)
        test_set_trial_ind = np.hstack(
            [np.repeat(n, len(test_set[n])) for n in range(test_set_num_trials)])
        if not T:
            T = np.max(test_set_flat)

        test_set_padded_mat = np.atleast_2d(np.hstack(
            [np.repeat(n, test_set[n].shape[0]) for n in range(test_set_num_trials)]))

        integration_points = np.repeat(self.integration_points[0][np.newaxis, :], test_set_num_trials, axis=0)
        integration_points_flat = np.hstack(integration_points)
        integration_points_trials_inds = np.hstack(
            [np.repeat(n, integration_points[n].shape[0]) for n in range(test_set_num_trials)])

        return integration_points_flat, test_set_flat, \
               integration_points_trials_inds, test_set_trial_ind, \
               test_set_padded_mat, test_set_num_trials, T

    def calc_posterior_for_ll(self, all_points, all_trial_inds, test_set_padded_mat):
        """
        This method computes the posterior for the lo-likelihood over the test set
        :param all_points: train data and test data
        :param all_trial_inds: indices of the trials of all the observations
        :param test_set_padded_mat: test set in a padded matrix form
        :return: the posterior mean and the cholesky decomposition of the posterior covariance
        """
        print('preparing kxx')
        Kxx_effects = self.jit_phi_kernel_effects(all_points, all_points,
                                                  all_trial_inds, all_trial_inds,
                                                  test_set_padded_mat, test_set_padded_mat,
                                                  self.hyper_params[:3])
        Kxx_background = self.jit_phi_kernel_background(all_points, all_points,
                                                        self.hyper_params[3:])
        Kxx = Kxx_background + Kxx_effects
        print('kxx ready')

        Ksx_effects = self.jit_phi_kernel_effects(all_points, self.induced_points_flat,
                                                  all_trial_inds,
                                                  self.induced_points_trials_inds,
                                                  test_set_padded_mat, self.observations_padded_mat,
                                                  self.hyper_params[:3])
        Ksx_background = self.jit_phi_kernel_background(all_points, self.induced_points_flat,
                                                        self.hyper_params[3:])
        Ksx = Ksx_background + Ksx_effects

        print('ksx ready')

        Kss_inv = self.Kss_inv

        print('done with Kss')
        kappa = Kss_inv.dot(Ksx)
        print('preparing Sigma post')
        Sigma_post = Kxx - kappa.T.dot(Ksx - self.Sigma_g_s.dot(kappa))
        print('preparing mu post')
        mu_post = kappa.T.dot(self.mu_g_s)
        print('preparing L post')
        L_post = np.linalg.cholesky(Sigma_post + self.noise * np.eye(
            Sigma_post.shape[0]))

        return mu_post, L_post

    def loglikelihood_test_data(self, test_set, num_samples=1e4, T=0):
        """
        This method estimates the log-likelihood over test data
        :param test_set: test data set. List of events array. len(test_set)=num_trials
        :param num_samples: how many samples from the posterior distribution to take for the log-likelihood estimation
        :param T: time bound of the test data
        :return: log-likelihood and intensities for each sample from the posterior
        """

        # TODO: add reference to equation from the paper

        print('preparing the data')
        integration_points_flat, test_set_flat, \
        integration_points_trials_inds, test_set_trial_ind, \
        test_set_padded_mat, test_set_num_trials, T = self.prepare_test_data_for_ll(test_set, T)
        print('done preparing data')

        total_num_integration_points = len(integration_points_flat)
        all_points = np.hstack([integration_points_flat, test_set_flat])
        all_trial_inds = np.hstack([integration_points_trials_inds, test_set_trial_ind])
        num_points = len(all_points)

        print('preparing posterior')
        mu_post, L_post = self.calc_posterior_for_ll(all_points, all_trial_inds, test_set_padded_mat)
        print('done preparing kernels')

        batch_size = 100
        num_iters = int(num_samples / batch_size)
        ll = onp.empty(int(num_samples))
        all_sampled_phis = []
        intensities = []
        for i in range(num_iters):
            print(i)
            # smaple from the posterior
            rand_nums = random.normal(self.key, (num_points, batch_size))
            sampled_phis = mu_post[:, np.newaxis] + L_post.dot(rand_nums)
            all_sampled_phis.append(sampled_phis)
            sampled_lambda = onp.random.gamma(shape=self.alpha_q1,
                                              scale=1. / self.beta_q1, size=batch_size) / test_set_num_trials
            sampled_intensity = sampled_lambda / (1. + np.exp(- sampled_phis))

            integral = - np.mean(sampled_intensity[:total_num_integration_points], axis=0) * T
            data_points_sum_log = np.sum(np.log(sampled_intensity[total_num_integration_points:]), axis=0)
            ll[i * batch_size: (i + 1) * batch_size] = integral + data_points_sum_log
            intensities.append(sampled_intensity)

        return ll, intensities

    def estimate_s_g_mean(self, grid_points):
        """
        This method estimate the mean of the background gp \s and the effects gp \g from the inferred intensity on a grid.
        :param grid_points: grid on which \s and \g are estimated - 1D array
        :return: the mean of \s (1D array) and \g (1D array)
        """

        # TODO - add reference to the equations from the paper.

        mu_tilde = np.dot(self.Kss_inv, self.mu_g_s)
        K_background = self.jit_phi_kernel_background(grid_points, self.induced_points_flat,
                                                      self.hyper_params[3:])
        s = np.dot(K_background.T, mu_tilde)

        K_effects = self.jit_kernel_with_decay_for_estimate_g(grid_points, self.induced_points_flat,
                                                              self.induced_points_trials_inds,
                                                              self.observations_padded_mat,
                                                              self.hyper_params[:3])
        g = np.dot(K_effects.T, mu_tilde)

        return s, g

    def estimate_s_g_variance(self, grid_points):
        """
        This method estimate the covariance of the background gp \s and the effects gp \g from the inferred intensity on a grid.
        :param grid_points: grid on which \s and \g are estimated - 1D array
        :return: the covariance of \s (2D array) and \g (2D array)
        """

        # TODO - add reference to the equations from the paper.

        B = self.Kss_inv - self.Kss_inv @ self.Sigma_g_s @ self.Kss_inv.T

        K_background = self.jit_phi_kernel_background(grid_points, grid_points,
                                                      self.hyper_params[3:])

        K_effects = self.jit_kernel_with_decay_no_history_for_estimate_g(grid_points, grid_points,
                                                                         self.hyper_params[:3])

        K_xs_background = self.jit_phi_kernel_background(grid_points, self.induced_points_flat,
                                                         self.hyper_params[3:])
        K_xs_effects = self.jit_kernel_with_decay_for_estimate_g(grid_points, self.induced_points_flat,
                                                                 self.induced_points_trials_inds,
                                                                 self.observations_padded_mat,
                                                                 self.hyper_params[:3])

        cov_s = K_background - K_xs_background.T @ B @ K_xs_background
        cov_g = K_effects - K_xs_effects.T @ B @ K_xs_effects

        return cov_s, cov_g
