import itertools
import time

import numpy as np
import sklearn
from .util import chunks
import matplotlib.pyplot as plt


def calculate_auroc(a_k, test_data, predictions, plot=False):
	assert len(a_k.shape) == 2
	threshold_count = 31
	thresholds = np.linspace(start=0, stop=1, num=threshold_count)
	digits = int(test_data.shape[1] / 2)

	assert test_data.shape[0] == len(predictions)
	Tones_obs = np.zeros(thresholds.size - 1)
	Tzeroes_obs = np.zeros(thresholds.size - 1)

	total_ones = 0
	total_zeroes = 0
	for n in range(test_data.shape[0]):
		cluster_bottom = a_k[predictions[n], digits:]
		obs_bottom = test_data[n, digits:]
		total_ones += obs_bottom[np.where(obs_bottom == 1)].size
		total_zeroes += obs_bottom[np.where(obs_bottom == 0)].size
		for idx in range(len(thresholds) - 1):
			predictions_mask = np.where((cluster_bottom < thresholds[idx + 1]) & (cluster_bottom >= thresholds[idx]))
			selected_obs = obs_bottom[predictions_mask]

			Tones_obs[idx] += selected_obs[np.where(selected_obs == 1)].size
			Tzeroes_obs[idx] += selected_obs[np.where(selected_obs == 0)].size

	pdf_ones = Tones_obs / total_ones
	pdf_zeroes = Tzeroes_obs / total_zeroes
	cdf_ones = np.cumsum(pdf_ones)
	cdf_zeroes = np.cumsum(pdf_zeroes)
	centers = (thresholds[1:] + thresholds[:-1]) / 2

	if plot:
		ax = plt.gca()
		plt.ylim((0.0, 1.0))
		plt.xlim((0.0, 1.0))
		plt.xlabel("Cumulative Proportion of Ones")
		plt.ylabel("Cumulative Proportion of Zeroes")
		plt.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
		plt.plot(cdf_ones, cdf_zeroes, marker="x", markersize=6, linewidth=3, color="blue")
		plt.fill_between(cdf_ones, 0, cdf_zeroes, alpha=0.2, color='blue')
		plt.title(f"Receiver Operating Characteristic")
		plt.grid(True, which="both")
		plt.show()
	return sklearn.metrics.auc(cdf_ones, cdf_zeroes)


def naive_align_labels(reference_params, estimated_params):
	K = reference_params.shape[0]
	D = reference_params.shape[1]
	permutations = itertools.permutations(range(K))
	best_permutation = None
	smallest_MSE = 10000

	start_time = time.time()
	for permutation in permutations:
		MSE = np.sum(np.power(reference_params - estimated_params[permutation, :], 2)) / (K * D)
		if MSE < smallest_MSE:
			best_permutation = permutation
			smallest_MSE = MSE

	return best_permutation, smallest_MSE


def fast_align_labels(reference_params, estimated_params):
	K = reference_params.shape[0]
	D = reference_params.shape[1]

	alignments = []
	for idx in range(K):
		mask = np.isin(np.arange(K), np.array(alignments), invert=True)

		selected = estimated_params[mask][
			np.argmin(np.sum(np.power(reference_params[idx, :] - estimated_params[mask], 2), axis=1))]
		for idx, row in enumerate(estimated_params):
			if np.all(row == selected):
				alignments.append(idx)
				break
	MSE = np.sum(np.power(reference_params - estimated_params[alignments, :], 2)) / (K * D)
	return alignments, MSE


def find_clusters(a_k, data_obs, alignments=None, partial=False):
	assert len(a_k.shape) == 2
	assert a_k.shape[0] < a_k.shape[1]

	digits = int(data_obs.shape[1])
	if partial:
		digits = int(data_obs.shape[1] / 2)

	labels = []
	if alignments is None:
		alignments = np.arange(a_k.shape[0])
	realigned_ak = a_k[alignments, :]
	if len(data_obs.shape) > 1:
		for n in range(data_obs.shape[0]):
			labels.append(np.argmin(np.power(data_obs[n, :digits] - realigned_ak[:, :digits], 2).sum(axis=1)))
	else:
		labels.append(np.argmin(np.power(data_obs[:digits] - realigned_ak[:, :digits], 2).sum(axis=1)))
	return labels


def average_of_estimates(input, size=10):
	assert (float(len(input)) / float(size)).is_integer()

	groups = chunks(input, size)
	means = []
	for idx, chunk in enumerate(groups):
		ref_params = chunk[0]
		aligned_params = [chunk[0]]

		for estimate in chunk[1:]:
			aligned_params.append(estimate[fast_align_labels(ref_params, estimate)[0], :])
		means.append(np.mean(np.stack(aligned_params, axis=0), axis=0))

	return means
