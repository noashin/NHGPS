import os
import sys
import time
import pickle
from shutil import copyfile
import traceback

sys.path.append('../../')

import jax
import numpy as np
from jax import config as config_jax
import click
import yaml

from src.variational_inference import VI


@click.command()
@click.option('--yml_file', type=click.STRING,
              help='path to the yaml file with the input')
@click.option('--output_path', type=click.STRING, default='./')
def main(yml_file, output_path):
    # configure jax
    config_jax.update("jax_enable_x64", True)
    config_jax.update("jax_debug_nans", True)

    time_stamp = time.strftime("%Y%m%d-%H%M%S")

    # read the input file
    with open(yml_file, 'r') as stream:
        config = yaml.safe_load(stream)
        settings_yml_file_path = config.get('settings_path', './')
        data_path = config.get('data_path', './')
        inference_results_path = config.get('inference_results_path', './')
        syn_data = config.get('syn_data', False)

    output_folder = os.path.join(output_path, f'{time_stamp}')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(settings_yml_file_path, 'r') as stream:
        config_data = yaml.safe_load(stream)

        num_inducing_points = config_data.get('num_inducing_points', 100)
        num_integration_points = config_data.get('num_integration_points', 1000)
        time_bound = config_data.get('time_bound', 1.)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    if syn_data:
        observations = data[0]
    else:
        observations = data

    with open(os.path.join(output_folder, 'input.yml'), 'w') as f:
        yaml.dump(config, f)
    with open(os.path.join(output_folder, 'data_input.yml'), 'w') as f:
        yaml.dump(config_data, f)
    with open(os.path.join(output_folder, 'data.p'), 'wb') as f:
        pickle.dump(data, f)

    hawkes_vi_object = VI(time_bound,
                          [],
                          num_inducing_points,
                          num_integration_points=num_integration_points)

    hawkes_vi_object.set_data(observations)

    with open(inference_results_path, 'rb') as f:
        inference_results = pickle.load(f)

    hawkes_vi_object.LB_list, hawkes_vi_object.mu_g_X, hawkes_vi_object.mu_g2_X, hawkes_vi_object.hyper_params_list, \
    hawkes_vi_object.induced_points, hawkes_vi_object.integration_points, hawkes_vi_object.Kss_inv, \
    hawkes_vi_object.ks_int_points, hawkes_vi_object.ks_X, hawkes_vi_object.observations, \
    hawkes_vi_object.Sigma_g_s, hawkes_vi_object.mu_g_s, hawkes_vi_object.lmbda_star_q1, \
    hawkes_vi_object.alpha_q1, hawkes_vi_object.beta_q1 = inference_results

    with open(os.path.join(output_folder, 'inference_results.p'), 'wb') as f:
        pickle.dump(inference_results, f)

    hawkes_vi_object.hyper_params = hawkes_vi_object.hyper_params_list[-1]
    hawkes_vi_object.induced_points_flat = np.hstack(hawkes_vi_object.induced_points)
    if not hasattr(hawkes_vi_object, 'induced_points_trials_inds'):
        hawkes_vi_object.induced_points_trials_inds = np.hstack(
            [np.repeat(n, hawkes_vi_object.induced_points[n].shape[0]) for n in range(hawkes_vi_object.num_trials)])

    grid = np.arange(0, time_bound, time_bound / 1000.)
    s, g = hawkes_vi_object.estimate_s_g_mean(grid)
    cov_s, cov_g = hawkes_vi_object.estimate_s_g_variance(grid)
    res_file = os.path.join(output_folder, 's_g_res.p')
    with open(res_file, 'wb') as f:
        pickle.dump([grid, s, g, cov_s, cov_g], f)


if __name__ == '__main__':
    main()
