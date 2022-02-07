import numpy as np
from scipy import interpolate

from .gps_jax import sample_gp, rbf_kernel


class NHGPS():
    """
    This class is used for data generation from the NH-GPS model.
    """

    def __init__(self, intensity_bound, time_bound, hypers, num_trials=1):
        """
        This method initializes the model object.
        :param intensity_bound: intensity bound of the model
        :param time_bound: the time bound of the model
        :param hypers: [effects_gp_output_variance, effects_gp_length_scale, memory_decay_factor, background_gp_output_variance, background_gp_length_scale]
        :param num_types: number of types of events n
        :param num_trials: number of trials
        """

        self.num_trials = num_trials
        self.time_bound = time_bound
        self.intensity_bound = intensity_bound  # array of length num_types

        self.hypers = hypers

        self.temporal_gp_sample = []
        self.temporal_gp_points = []
        self.temporal_gp_interpolated_points = []
        self.temporal_gp_interpolated_values = []

        self.temporal_gp_interpolated_background_points = []
        self.temporal_gp_interpolated_background_values = []
        self.temporal_gp_background_sample = []
        self.temporal_gp_background_points = []

        self.intensities = []
        self.phis = []
        self.self_effects = []

    def reset_intensities(self):
        """
        This method resets the intensities, phis and self effects
        """
        self.intensities = []
        self.phis = []
        self.self_effects = []

    def generate_temporal_gp(self, grid_points):
        """
        This method generates a 1d gp sample given 1d data vector, and stores it as a class attribute.
        As the training data for the gp, the method takes all the differences between the data points.
        :param grid_points: vector of shape n x 1.
        :return:
        """

        kernel = rbf_kernel(grid_points, grid_points, self.hypers[0], self.hypers[1])
        self.kk = kernel
        gp_sample = sample_gp(kernel)
        self.temporal_gp_sample = gp_sample
        self.temporal_gp_points = grid_points

        kernel_background = rbf_kernel(grid_points, grid_points, self.hypers[3], self.hypers[4])
        gp_sample = sample_gp(kernel_background)
        self.temporal_gp_background_sample = gp_sample
        self.temporal_gp_background_points = grid_points

        return

    def calculate_self_effects(self, candidate_point, history):
        """
        This method calculates the self effects given the history at candidate_point
        :param candidate_point: candidate event
        :param history: accepted events until candidate_point
        :param type_index: type of the event
        :return: the value of the self effects at candidate_point
        """
        memory_decay = self.hypers[2]
        self_effects = 0.
        assert self.temporal_gp_points is not None or self.temporal_gp_sample is not None, "Please generate a temporal gp"

        history = np.array(history)
        if len(history) > 0:
            relevant_history = history[np.where(candidate_point > history)]
            for history_point in relevant_history:
                time_difference = candidate_point - history_point
                self_effects += self.temporal_gp_function(time_difference) * np.exp(- memory_decay * time_difference)

            return self_effects
        else:
            return 0

    def get_background_intensity(self, candidate_point):
        return self.temporal_gp_function(candidate_point, for_background=True)

    def temporal_gp_function(self, value, for_background=False):
        """
        This method evaluates the gp at a certain point (from the gp's training data).
        :param value: point where to evaluate the gp.
        :param for_background: if true the gp function of the background rate is evaluated
        :return: gp's value.
        """

        # use the correct GP - the background GP or the self effects GP
        temporal_gp_interpolated_points = self.temporal_gp_interpolated_background_points if for_background else \
            self.temporal_gp_interpolated_points
        temporal_gp_interpolated_values = self.temporal_gp_interpolated_background_values if for_background else \
            self.temporal_gp_interpolated_values
        temporal_gp_sample = self.temporal_gp_background_sample if for_background else self.temporal_gp_sample
        temporal_gp_points = self.temporal_gp_background_points if for_background else self.temporal_gp_points

        # Every time we evaluate the gp in a point we check if we already evaluated the gp there.
        # If yes, we look for it in the map, if not we evaluate the gp and store
        # the point and the gp value.

        temporal_gp_interpolated_points_arr = np.array(temporal_gp_interpolated_points)
        ind = np.argwhere(temporal_gp_interpolated_points_arr == value)
        if len(ind):
            return temporal_gp_interpolated_values[ind[0][0]]
        else:
            temporal_gp_interpolated_points.append(value)
            temporal_gp_value = get_gp_value_for_sample(value, temporal_gp_sample,
                                                        temporal_gp_points)
            temporal_gp_interpolated_values.append(temporal_gp_value)

            return temporal_gp_value

    def sort_candidates(self, candidates):
        return np.sort(candidates.flatten())

    def thinning(self, candidates, use_history=True):
        """
        This method implements the thinning algorithm for Hawkes process and generates data from the model-
        :param candidates: Candidate points matrix num_trials x num_types x n
        :return: accepted data points matrix num_trials x num_types x m
        """

        history = []
        for k, candidate_trial in enumerate(candidates):
            print(k)
            history.append([])
            self.phis.append([])
            self.self_effects.append([])
            self.intensities.append([])
            for i, candidate in enumerate(candidate_trial):
                intensity = self.evaluate_intensity(candidate, history[-1], use_history)
                r = np.random.uniform()
                if r < intensity / self.intensity_bound:
                    history[-1].append(candidate)
                    self.intensities[-1].append(intensity)
                else:
                    self.phis[-1] = self.phis[-1][:-1]
                    self.self_effects[-1] = self.self_effects[-1][:-1]
            history[-1] = np.array(history[-1])
        return history

    def evaluate_intensity(self, candidate_point, history, use_history=True):
        """
        This method evaluates the Hawkes process intensity at a certain point give the history.
        :param candiate_point: the data point in which the intensity should be evaluated (x,y,t)
        :param history: the history of the process [(x_i,y_i,t_i)] t_{i-1} < t_i
        :param self_effects: Boolean. If False than the intensity is of a Poisson process.
        :return: the intensity
        """

        background_rate = self.get_background_intensity(candidate_point)
        self_effects = 0.

        if use_history:
            self_effects = self.calculate_self_effects(candidate_point, history)

        linear_intensity = background_rate + self_effects
        self.phis[-1].append(linear_intensity)
        self.self_effects[-1].append(self_effects)

        intensity = self.intensity_bound / (1. + np.exp(- linear_intensity))
        return intensity

    def generate_candidates(self, number_of_candidates, num_trials):
        """
        This method generates candidates for the thinning the process from a uniform distribution.
        :param number_of_candidates: number of candidates to be generated.
        :return: a matrix n x d of data points.
        """
        candidates = [np.random.uniform(0, self.time_bound, number_of_candidates[i]) for i in
                      range(num_trials)]
        sorted_candidates = [self.sort_candidates(candidates[i]) for i in range(num_trials)]

        return sorted_candidates


def get_gp_value_for_sample(sample, gp, gp_points):
    """
    This function evaluautes the value of the gp at sample using interpolation.
    :param sample: point where to evaluate the gp
    :param gp: gp values
    :param gp_points: points where the gp is already evaluated.
    :return: value of the gp at sample.
    """

    try:
        gp_np = gp.data.numpy()
        gp_points_np = gp_points.data.numpy()
    except AttributeError:
        # import jax.numpy as np
        gp_np = gp
        gp_points_np = gp_points
    if np.array(sample).shape:
        ind = np.argwhere(np.equal(gp_points_np, sample).sum(axis=1) == gp_points_np.shape[1])
        # if the point where we want to estimate the gp is in the gp input- return the gp value at this point
        # else, perform spline interpolation.
        if len(ind):
            return gp_np[ind[0]]
        else:
            tck = interpolate.bisplrep(gp_points_np[:, 0], gp_points_np[:, 1], gp_np)
            interpolated_value = interpolate.bisplev(sample[0], sample[1], tck)
            return interpolated_value
    else:
        ind = np.argwhere(gp_points_np == sample)
        # if the point where we want to estimate the gp is in the gp input- return the gp value at this point
        # else, perform spline interpolation.
        if len(ind):
            return gp_np[ind[0]]
        else:
            s = interpolate.InterpolatedUnivariateSpline(gp_points_np, gp_np)
            interpolated_value = s(sample)
            return interpolated_value
