import os
import sys
import time
import pickle

sys.path.append('../../')

import jax
from jax import config as config_jax
import click
import yaml

from src.variational_inference import VI


@click.command()
@click.option('--yml_file', type=click.STRING,
              help='path to the yaml file with the input', default='./')
@click.option('--output_path', type=click.STRING, default='./',
              help='destination path for the inference results')
def main(yml_file, output_path):
    # JAX configuration
    config_jax.update("jax_enable_x64", True)
    config_jax.update("jax_debug_nans", True)

    print(jax.devices())
    time_stamp = time.strftime("%Y%m%d-%H%M%S")

    # read the input file with the paths
    with open(yml_file, 'r') as stream:
        config = yaml.safe_load(stream)
        settings_yml_file_path = config.get('data_input', './')
        data_path = config.get('data', './')
        syn_data = config.get('syn_data', True)

    # where to save the inference results
    output_folder = os.path.join(output_path, f'{time_stamp}')
    print(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # read the settings file
    print(f'reading the following settings file: {settings_yml_file_path}')
    with open(settings_yml_file_path, 'r') as stream:
        config_data = yaml.safe_load(stream)

        lambda_a_prior = config_data.get('lambda_a_prior', 2.)
        lambda_b_prior = config_data.get('lambda_b_prior', 0.03)
        infer_max_intensity = config_data.get('infer_max_intensity', True)
        save_steps = config_data.get('save_steps', True)

        infer_hypers = config_data.get('infer_hypers', False)
        hyper_updates = config_data.get('hyper_updates', 1)
        init_effects_kernel_output_variance = config_data.get('init_effects_kernel_output_variance', 1.)
        init_effects_kernel_length_scale = config_data.get('init_effects_kernel_length_scale', 1.)
        init_memory_decay = config_data.get('init_memory_decay', 1.)
        init_backgroung_kernel_length_scale = config_data.get('init_backgroung_kernel_length_scale', 1.)
        init_background_kernel_output_variance = config_data.get('init_background_kernel_output_variance', 1.)

        effects_kernel_output_variance = config_data.get('effects_kernel_output_variance', 1.)
        effects_kernel_length_scale = config_data.get('effects_kernel_length_scale', 1.)
        memory_decay = config_data.get('memory_decay', 1.)
        backgroung_kernel_length_scale = config_data.get('backgroung_kernel_length_scale', 1.)
        background_kernel_output_variance = config_data.get('background_kernel_output_variance', 1.)

        grad_step_size = config_data.get('grad_step_size', 0.01)
        adapt_grad_step_size = config_data.get('adapt_grad_step_size', False)

        num_inducing_points = config_data.get('num_inducing_points', 100)
        num_integration_points = config_data.get('num_integration_points', 1000)
        convergence_criteria = config_data.get('convergence_criteria', 1e-4)
        noise = config_data.get('noise', 1e-4)
        time_bound = config_data.get('time_bound', 1.)

    # save everything in the results folder
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

    # set the hyper parameters to the ground truth if they are not to be inferred
    if syn_data and not infer_hypers:
        hypers = [effects_kernel_output_variance, effects_kernel_length_scale, memory_decay,
                  backgroung_kernel_length_scale, background_kernel_output_variance]
    else:
        hypers = [init_effects_kernel_output_variance, init_effects_kernel_length_scale, init_memory_decay,
                  init_backgroung_kernel_length_scale, init_background_kernel_output_variance]

    file_name = 'inference_res.p'
    file_path = os.path.join(output_folder, file_name)
    hawkes_vi_object = VI(time_bound,
                          hypers,
                          num_inducing_points,
                          alpha_0=lambda_a_prior,
                          betta_0=lambda_b_prior,
                          lmbda_star=None,
                          conv_crit=convergence_criteria,
                          num_integration_points=num_integration_points,
                          noise=noise)

    hawkes_vi_object.set_data(observations)

    hawkes_vi_object.run(save_steps=save_steps, file_path=file_path, hyper_parms_inference=infer_hypers,
                         infer_max_intensity=infer_max_intensity, grad_step_size=grad_step_size,
                         adapt_grad_step_size=adapt_grad_step_size,
                         hyper_updates=hyper_updates,
                         output=True)


if __name__ == '__main__':
    main()
