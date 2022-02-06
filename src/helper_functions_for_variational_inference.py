import jax.numpy as np
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import digamma, gammaln


def calculate_postrior_GP(lmbda_q2, mu_omega_int_points, kappa_X, mu_omega_X,
                          kappa_int_points, num_integration_points, T, Kss_inv, noise,
                          ks_X, ks_int_points):
    """
    This method calculates the mean and the covariance of the GP in the inducing points.
    :param lmbda_q2: rate of the augmenting observations
    :param mu_omega_int_points: the mean of the GP variables that correspond to the integration points.
    :param kappa_X: Kss_inv * ks_X
    :param mu_omega_X: the mean of the GP variables that correspond to the observations.
    :param kappa_int_points: Kss_inv * ks_int_points
    :param num_integration_points: the total number of integration points
    :param T: time bound
    :param Kss_inv: inverse of the kernel of the inducing points
    :param noise: how much noise to add to the diagonal
    :param ks_X: kernel between the inducing points and the observations
    :param ks_int_points: kernel between the inducing points and the integration points
    :return: covariance, inverse of the covariance, log-determinant of the covariance and the mean
    """

    # TODO: add reference to equation on the paper

    A_int_points = lmbda_q2 * mu_omega_int_points
    kAk = kappa_X.dot(mu_omega_X[:, np.newaxis] * kappa_X.T) + kappa_int_points.dot(A_int_points[:, np.newaxis] *
                                                                                    kappa_int_points.T) / num_integration_points * T
    Sigma_g_s_inv = kAk + Kss_inv
    L_inv = np.linalg.cholesky(Sigma_g_s_inv + noise * np.eye(Sigma_g_s_inv.shape[0], dtype=np.float64))
    L = solve_triangular(L_inv, np.eye(L_inv.shape[0], dtype=np.float64), lower=True,
                         check_finite=False)
    Sigma_g_s = L.T.dot(L)
    logdet_Sigma_g_s = 2 * np.sum(np.log(L.diagonal()))
    b_int_points = -.5 * lmbda_q2
    b_X = .5 * np.ones(ks_X.shape[1], dtype=np.float64)
    kb = ks_X.dot(b_X) + ks_int_points.dot(b_int_points) / \
         num_integration_points * T
    mu_g_s = Sigma_g_s.dot(kb.dot(Kss_inv))

    return Sigma_g_s_inv, Sigma_g_s, logdet_Sigma_g_s, mu_g_s


def predictive_posterior_GP(ks_x_prime, kappa, mu_g_s, Sigma_g_s, k_xprime_xprime_diag):
    """
    This function computes the predictive posterior GP over points x conditioned on the GP
    In the inducing points.
    :param ks_x_prime: kernel between the inducing points and tc.
    :param kappa: Kss_inv * ks_x
    :param mu_g_s: mean of the GP in the inducing points.
    :param Sigma_g_s: covariance of the GP in the inducing points,
    :param k_xprime_xprime_diag: diagonal of the kernel of x.
    :return: mean and variance of the GP in points x.
    """

    # TODO: add reference to equation on the paper

    mu_g_x_prime = kappa.T.dot(mu_g_s)
    var_g_x_prime = k_xprime_xprime_diag - np.sum(kappa * (ks_x_prime - kappa.T.dot(Sigma_g_s).T), axis=0)
    return mu_g_x_prime, var_g_x_prime


def calculate_lower_bound(Sigma_g_s, mu_g_s, mu_g_int_points, mu_g2_int_points, mu_omega_int_points,
                          lmbda_q2, c_int_points, log_lmbda_star_q1, mu_g_X, mu_g2_X, mu_omega_X, c_X,
                          num_integration_points, T, lmbda_star_q1, Kss_inv, logdet_Ks, logdet_Sigma_g_s,
                          num_inducing_points, alpha0, beta0, alpha_q1, beta_q1):
    """
    This methods calculates the lower bound.
    :param Sigma_g_s: covariannce of the GP in the inducing points.
    :param mu_g_s: mean of the GP in the inducing points.
    :param mu_g_int_points: mean of the GP in the integration points.
    :param mu_g2_int_points: second moment of the GP in the integration points.
    :param mu_omega_int_points: mean of the PG variables that correspond to the integration points.
    :param lmbda_q2: rate of the augmenting observations.
    :param c_int_points: square root of second moment of the GP in the integration points.
    :param log_lmbda_star_q1: log of the mean of intensity bound.
    :param mu_g_X: mean of the GP in the observations.
    :param mu_g2_X: second moment of the GP in the observations.
    :param mu_omega_X: mean of the PG variables that correspond to the observations.
    :param c_X: square root of second moment of the GP in the observations.
    :param num_integration_points: total number of integration points.
    :param T: time bound
    :param lmbda_star_q1: mean of the intensity bound
    :param Kss_inv: inverse of the kernel of the inducing points.
    :param logdet_Ks: log determinant of the kernel of the inducing points.
    :param logdet_Sigma_g_s: log determinant of the covariance of the inducing points.
    :param num_inducing_points: total number of inducing points.
    :param alpha0: alpha value of the prior over the intensity bound
    :param beta0: beta value of the prior over the intensity bound
    :param alpha_q1: alpha value of the posterior over the intensity bound
    :param beta_q1: beta value of the posterior over the intensity bound
    :return: the variational lower bound.
    """
    Sigma_s_mugmug = Sigma_g_s + np.outer(mu_g_s, mu_g_s)
    f_int_points = .5 * (- mu_g_int_points -
                         mu_g2_int_points * mu_omega_int_points) - \
                   np.log(2.)
    integrand = f_int_points - \
                np.log(lmbda_q2 * np.cosh(.5 * c_int_points)) \
                + log_lmbda_star_q1 + \
                .5 * c_int_points ** 2 * mu_omega_int_points + 1.
    f_X = .5 * (mu_g_X - mu_g2_X * mu_omega_X) - \
          np.log(2.)
    summand = f_X + log_lmbda_star_q1 - np.log(np.cosh(
        .5 * c_X)) + .5 * c_X ** 2 * mu_omega_X
    L = integrand.dot(lmbda_q2) / num_integration_points * T
    L -= lmbda_star_q1 * T
    L += np.sum(summand)
    L -= .5 * np.trace(Kss_inv.dot(Sigma_s_mugmug), dtype=np.float64)
    L -= .5 * logdet_Ks
    L += .5 * logdet_Sigma_g_s + .5 * num_inducing_points
    L += alpha0 * np.log(beta0) - gammaln(alpha0) + \
         (alpha0 - 1) * log_lmbda_star_q1 - \
         beta0 * lmbda_star_q1
    L += alpha_q1 - np.log(beta_q1) + gammaln(alpha_q1) \
         + (1. - alpha_q1) * digamma(alpha_q1)

    return L


def partial_derivative(dKs, dks_X, dks_int_points, dKxx_diag, dKii_diag, Kss_inv, ks_int_points, kappa_X, ks_X,
                       kappa_int_points, mu_g_s, Sigma_s_mugmug, mu_omega_X, mu_omega_int_points,
                       lmbda_q2, num_integration_points, T):
    """
    This function calculates the Jacobian of the ELBO with respect to the model hyper parameters
    :param dKs: partial derivatives of the kernels over the induced points
    :param dks_X: partial derivatives of the kernels over the real data
    :param dks_int_points: partial derivatives of the kernels over the integration points
    :param dKxx_diag: partial derivatives of the diagonal of the kernels over the real data
    :param dKii_diag: partial derivatives of the diagonal of the kernels over the integration points
    :param Kss_inv: inverse of the kernel over the induced points
    :param ks_int_points: kernel between the induced and integration points
    :param kappa_X: Kss_inv * ks_X
    :param ks_X: kernel between the induced points and the real data
    :param kappa_int_points: Kss_inv * ks_int_points
    :param mu_g_s: mean of the variational distribution in the inducing points
    :param Sigma_s_mugmug:
    :param mu_omega_X: mean of the variational distribution over the augmenting PG variables with respect to the observations.
    :param mu_omega_int_points: mean of the variational distribution over the augmenting PG variables with respect to the integration points.
    :param lmbda_q2: The distribution over the augmenting events.
    :param num_integration_points: number of integration points
    :param T: time bound of the data
    :return: JAcobian of the partial derivatives. array
    """

    dKs_inv = - Kss_inv.dot(dKs.dot(Kss_inv))

    dkappa_X = Kss_inv.dot(dks_X) + dKs_inv.dot(ks_X)
    dkappa_int_points = Kss_inv.dot(dks_int_points) + dKs_inv.dot(ks_int_points)

    dKtilde_X = dKxx_diag - np.sum(dks_X * kappa_X, axis=0) - np.sum(ks_X * dkappa_X, axis=0)
    dKtilde_int_points = dKii_diag - np.sum(dks_int_points * kappa_int_points, axis=0) - np.sum(
        ks_int_points * dkappa_int_points, axis=0)

    dg1_X = mu_g_s.dot(dkappa_X)
    dg1_int_points = mu_g_s.dot(dkappa_int_points)

    dg2_X = (dKtilde_X + 2. * np.sum(kappa_X * Sigma_s_mugmug.dot(dkappa_X), axis=0)) * mu_omega_X
    dg2_int_points = (dKtilde_int_points + 2. * np.sum(kappa_int_points * Sigma_s_mugmug.dot(dkappa_int_points),
                                                       axis=0)) * mu_omega_int_points

    dL_dtheta_tmp = .5 * (np.sum(dg1_X) - np.sum(dg2_X))
    dL_dtheta_tmp += .5 * np.dot(
        - dg1_int_points - dg2_int_points, lmbda_q2) / num_integration_points * T
    dL_dtheta_tmp -= .5 * np.trace(Kss_inv.dot(dKs))
    dL_dtheta_tmp += .5 * np.trace(Kss_inv.dot(dKs.dot(Kss_inv.dot(Sigma_s_mugmug))))

    return dL_dtheta_tmp
