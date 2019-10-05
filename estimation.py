import pickle
import pickle
import time

from matplotlib import rcParams
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from .gibbs import BayesMixture, find_clusters, BernoulliMixtureEM
from sklearn.metrics import accuracy_score
from .metrics import calculate_auroc, fast_align_labels
import h5py
from .util import get_digits, chunks, kde_sklearn

rcParams['figure.figsize'] = 6, 5


def test_estimation():
	digits, labels = get_digits(0.8)
	selected_digits = digits[np.where((labels == 6) | (labels == 1) | (labels == 3))]
	mixture = BayesMixture(selected_digits, K=3, alpha=300, beta=0.89, delta=0.5, seed=8000)

	z_new, a_k, gamma, mixing_props = mixture.gibbs_sample(max_iterations=52, hard_cluster=False)

	fig, axes = plt.subplots(1, mixture.K)
	for k in range(0, mixture.K):
		cax = axes[k].matshow(a_k[k, :, -1].reshape(28, 28))
	plt.show()


def hyperparameter_search():
	digits, labels = get_digits(0.8)
	condition = (labels == 1) | (labels == 6) | (labels == 9)
	selected_digits = digits[np.where(condition)]
	selected_labels = labels[np.where(condition)]
	X_train, X_test, y_train, y_test = train_test_split(selected_digits, selected_labels, test_size=0.1,
														random_state=42)
	iters = 1000

	reference_params = np.load("reference_params.npy")
	y_test[y_test == 1] = 0
	y_test[y_test == 6] = 1
	y_test[y_test == 9] = 2
	values = []

	parameters = {
		"alpha": {
			"min": 10,
			"max": 500
		},
		"beta": {
			"min": 0.1,
			"max": 0.9
		},
		"delta": {
			"min": 0.1,
			"max": 0.9
		}
	}
	for trial_idx in range(iters):
		start_time = time.time()
		alpha_sample = np.random.randint(parameters["alpha"]["min"], parameters["alpha"]["max"])
		beta_sample = np.random.rand() * (parameters["beta"]["max"] - parameters["beta"]["min"]) + parameters["beta"][
			"min"]
		delta_sample = np.random.rand() * (parameters["delta"]["max"] - parameters["delta"]["min"]) + \
					   parameters["delta"]["min"]
		iters = np.random.randint(30, 31)

		mixture = BayesMixture(X_train, K=3, alpha=alpha_sample, beta=beta_sample, delta=delta_sample, seed=42)
		print(
			f"Testing hyperparams with alpha={alpha_sample}, beta={beta_sample:.3f}, delta={delta_sample:.3f}, iters={iters}")
		z_new, a_k, gamma, mixing_props = mixture.gibbs_sample(max_iterations=iters, hard_cluster=False, verbose=False)
		alignments, _ = fast_align_labels(reference_params[:, :, -1], a_k[:, :, -1])

		predictions = mixture.find_clusters(X_test, alignments)
		score = accuracy_score(y_test, predictions, normalize=True)
		print(f"\tAccuracy: {score:.4f}. Took {time.time() - start_time:.3f}s")


# noinspection PyUnreachableCode
def test_auroc_variability_evaluation():
	# X_train, y_train, X_test , y_test = get_digits(0.5)

	simple = []
	mean = []
	if False:
		with h5py.File("4_estimates2.h5", "r") as h5:
			estimates = [h5[key][()] for key in h5.keys()]

			simple_sample = estimates[:100]
			mean_sample = chunks(estimates[100:], 10)

			for idx, chunk in enumerate(mean_sample):
				ref_params = chunk[0]
				aligned_params = [chunk[0]]

				for estimate in chunk[1:]:
					aligned_params.append(estimate[fast_align_labels(ref_params, estimate)[0], :])
				mean_params = np.mean(np.stack(aligned_params), axis=0)

				predictions = find_clusters(mean_params, X_test, partial=True)
				auroc = calculate_auroc(mean_params, X_test, predictions)
				mean.append(auroc)
				print(f"Done chunk {idx}")

			for idx, estimate in enumerate(simple_sample):
				predictions = find_clusters(estimate, X_test, partial=True)
				auroc = calculate_auroc(estimate, X_test, predictions)
				simple.append(auroc)
				print(f"Done chunk {idx}")

		with open("variance_comparison.bin", "wb") as f:
			pickle.dump({"simple": simple, "mean": mean}, f)

	with open("./datasets/variance_comparison.bin", "rb") as f:
		data = pickle.load(f)
	simple = data["simple"]
	mean = data["mean"]

	x_grid = np.linspace(0.88, 0.92, 1000)
	plt.ylim((0, 130))
	plt.fill(x_grid, kde_sklearn(simple, x_grid), alpha=0.5, label="Simple")
	plt.fill(x_grid, kde_sklearn(mean, x_grid), alpha=0.5, label="Averaged")
	plt.xlabel("AUROC")
	plt.ylabel("Kernel Density Estimate")
	plt.legend(loc="upper left")

	plt.show()


def test_auroc_single():
	digits, labels = get_digits(0.5)

	condition = (labels == 4)
	selected_digits = digits[np.where(condition)]
	selected_labels = labels[np.where(condition)]
	X_train, X_test, y_train, y_test = train_test_split(selected_digits, selected_labels, test_size=0.2,
														random_state=42)
	for idx in range(10):
		mixture = BayesMixture(X_train, K=8, alpha=50, beta=0.5, delta=0.8, seed=int(time.time()))
		z_new, a_k, gamma, mixing_props = mixture.gibbs_sample(max_iterations=30, hard_cluster=False, verbose=False)

		predictions = mixture.find_clusters(X_test, partial=True)
		auroc = calculate_auroc(mixture.a_k[:, :, -1], X_test, predictions)
		print(f"Bayesian: {auroc}")

		EMMixture = BernoulliMixtureEM(K=8, D=28 * 28, X=X_train)
		estimates, _ = EMMixture.train(max_iterations=40, display_interval=10, save_interval=None, verbose=False)
		estimates = estimates.T
		predictions = find_clusters(estimates, X_test, partial=True)
		auroc = calculate_auroc(estimates, X_test, predictions)
		print(f"EM: {auroc}")


def test_auroc_many():
	digits, labels = get_digits(0.5)

	for digit in [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]:
		condition = (labels == digit)
		selected_digits = digits[np.where(condition)]
		selected_labels = labels[np.where(condition)]
		X_train, X_test, y_train, y_test = train_test_split(selected_digits, selected_labels, test_size=0.2,
															random_state=42)
		auroc_list = []
		k_list = [2, 5, 8, 10, 15, 20, 30]
		for k_val in k_list:
			mixture = BayesMixture(X_train, K=k_val, alpha=50, beta=0.5, delta=0.8, seed=7001)
			z_new, a_k, gamma, mixing_props = mixture.gibbs_sample(max_iterations=30, hard_cluster=False, verbose=False)

			predictions = mixture.find_clusters(X_test, partial=True)

			auroc_list.append(calculate_auroc(mixture.a_k, X_test, predictions))
			print(f"Done {digit}-{k_val}")

		plt.plot(k_list, auroc_list, marker="x", label=f"{digit}")
	plt.legend(loc="upper right")
	plt.show()


def worked_example():
	digits, labels = get_digits(0.8)
	condition = (labels == 0) | (labels == 1)
	selected_digits = digits[np.where(condition)]
	selected_labels = labels[np.where(condition)]

	obs_inspected = [36, 40]
	digits_inspected = [425, 435]

	rcParams['figure.figsize'] = 7, 5
	fig, axes = plt.subplots(1, 2)
	for k in range(0, 2):
		cax = axes[k].matshow(selected_digits[obs_inspected[k]].reshape(28, 28))
		axes[k].plot(15, 15)
	plt.close()


	with open("plot_gamma.bin", "rb") as file:
		z_new, a_k, gamma, mixing_props = pickle.load(file)

	rcParams['figure.figsize'] = 7, 5
	fig, axes = plt.subplots(1, 2)
	for k in range(0, 2):
		cax = axes[k].matshow(a_k[k, :, -1].reshape(28, 28))
		axes[k].plot(15, 15)
	# plt.colorbar(cax)
	# plt.show()
	plt.close()

	rcParams['figure.figsize'] = 7, 5
	x_axis = np.arange(0, gamma.shape[-1] - 1)
	fig, axes = plt.subplots(nrows=1, ncols=2)
	for obs_idx in range(2):
		axes[obs_idx].set_title(f"Gamma: {obs_idx} digit")
		axes[obs_idx].set_ylim(bottom=-0.1, top=1.1)
		axes[obs_idx].plot(x_axis, gamma[obs_inspected[obs_idx], :, 1:].T)
		axes[obs_idx].legend(loc='center right', labels=["k=0", "k=1"])
		axes[obs_idx].grid(True, which="both")
	plt.show()
	plt.close()

	rcParams['figure.figsize'] = 8, 5
	x_axis = np.arange(0, z_new.shape[-1] - 1)
	fig, axes = plt.subplots(nrows=1, ncols=2)
	cluster_idx = np.argmax(z_new[obs_inspected, :, 1:], axis=1)
	for obs_idx in range(2):
		axes[obs_idx].set_title(f"z_n for {obs_idx} digit")
		axes[obs_idx].set_ylim(bottom=-0.1, top=1.1)
		axes[obs_idx].scatter(x_axis, cluster_idx[obs_idx, :].T)
		axes[obs_idx].grid(True, which="both")
	plt.close()

	rcParams['figure.figsize'] = 8, 5
	x_axis = np.arange(0, a_k.shape[-1] - 1)
	fig, axes = plt.subplots(nrows=1, ncols=2)
	for cluster_idx in range(2):
		axes[cluster_idx].set_title(f"a_k estimates: {'Left' if cluster_idx == 0 else 'Center'} Pixel ")
		axes[cluster_idx].set_ylim(bottom=-0.1, top=1.1)
		axes[cluster_idx].plot(x_axis, a_k[:, digits_inspected[cluster_idx], 1:].T)
		axes[cluster_idx].grid(True, which="both")
		axes[cluster_idx].legend(loc='upper left', labels=["0", "1"])
	plt.close()


if __name__ == "__main__":
	# test_estimation()
	# calculate_accuracy()
	# hyperparameter_search()
	# worked_example()
	test_estimation()
