import os
import sys
import time
import pickle

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
    config_jax.update("jax_enable_x64", True)
    config_jax.update("jax_debug_nans", True)

    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    output_folder = os.path.join(output_path, f'{time_stamp}')

    with open(yml_file, 'r') as stream:
        config = yaml.safe_load(stream)
        data_path = config.get('data_path', './')
        inference_results_path = config.get('inference_results_path', './')
        test_data_path = config.get('test_data_path', './')
        num_samples = config.get('num_samples', 1000)
        syn_data = config.get('syn_data', False)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    if syn_data:
        observations = data[0]
    else:
        observations = data

    with open(test_data_path, 'rb') as f:
        test_set_data = pickle.load(f)
    if syn_data:
        test_observations = test_set_data[0]
    else:
        test_observations = test_set_data

    with open(os.path.join(output_folder, 'input.yml'), 'w') as f:
        yaml.dump(config, f)


    with open(os.path.join(output_folder, 'data.p'), 'wb') as f:
        pickle.dump(data, f)

    with open(os.path.join(output_folder, 'test_data.p'), 'wb') as f:
        pickle.dump(test_set_data, f)

    hawkes_vi_object = VI(0, [], 0)
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

    ll, intensities = hawkes_vi_object.loglikelihood_test_data(test_observations,
                                                              num_samples=num_samples)
    ll_file = os.path.join(output_folder, 'test_ll_res.p')
    with open(ll_file, 'wb') as f:
        pickle.dump([ll, intensities], f)


if __name__ == '__main__':
    main()
