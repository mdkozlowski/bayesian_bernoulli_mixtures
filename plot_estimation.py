import pickle
import time

from matplotlib import rcParams
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
from .estimation import get_digits, calculate_auroc
from .gibbs import BayesMixture, find_clusters
from .metrics import fast_align_labels
from .util import kde_sklearn, chunks

rcParams['figure.figsize'] = 6, 5


def label_switching():
	def mutated(p):
		return np.clip(p + (np.random.random_sample((3, 3)) - 0.5) * 0.2, a_min=0, a_max=1)

	N_each = np.array([1000, 1000, 1000])
	p = np.array([
		[0, 0, 0, 0],
		[0, 0, 0, 0],
		[0, 0, 0, 0],
	])

	rcParams['figure.figsize'] = 5, 4
	mutated_p = np.stack([mutated(p), mutated(p), mutated(p)], axis=0)
	mutated_p[0] = mutated_p[0][[0, 1, 2], :]
	mutated_p[1] = mutated_p[1][[0, 2, 1], :]
	mutated_p[2] = mutated_p[2][[2, 0, 1], :]

	fig, axes = plt.subplots(1, 3)
	for k in range(0, 3):
		cax = axes[k % 3].matshow(mutated_p[k, :, :], vmin=0, vmax=1)
		axes[k % 3].set_xlabel("")
		axes[k % 3].set_ylabel("")
	plt.show()


def plot_gamma_heuristic():
	X_train, y_train, X_test, y_test = get_digits(0.5)
	X_train = X_train[np.where(y_train == 4)]
	mixture = BayesMixture(X_train, K=8, alpha=52, beta=0.5, delta=0.8, seed=int(time.time()))
	z_new, a_k, gamma, mixing_props = mixture.gibbs_sample(max_iterations=52, hard_cluster=False, verbose=True,
														   early_stopping=False)
	first_differences = (np.mean(np.abs(np.diff(gamma, axis=2, n=2)), axis=(0, 1)))
	plt.plot(first_differences)
	plt.axhline(y=0.003, color='r', alpha=0.8, ls="--")
	plt.title("First Differences for Average Gamma Parameters")
	plt.ylabel("Delta")
	plt.xlabel("Iterations")
	plt.show()


def plot_clusters():
	X_train, y_train, X_test, y_test = get_digits(0.5)
	selected_X = X_train[np.where((y_train == 4))]
	clusters = 5
	mixture = BayesMixture(selected_X, K=clusters, alpha=52, beta=0.5, delta=0.8, seed=7219)
	z_new, a_k, gamma, mixing_props, bad_ak = mixture.gibbs_sample(max_iterations=52, hard_cluster=False)

	rcParams['figure.figsize'] = clusters, 1
	fig, axes = plt.subplots(1, clusters)
	for k in range(0, clusters):
		cax = axes[k % clusters].imshow(a_k[k, :, -2].reshape(28, 28), vmin=0, vmax=1, interpolation="nearest",
										cmap='plasma')
		axes[k % clusters].set_xlabel("")
		axes[k % clusters].set_ylabel("")
		axes[k % clusters].axis("off")
	plt.tight_layout()
	plt.savefig("fill_in.png", dpi=600)


def plot_fill_in():
	X_train, y_train, X_test, y_test = get_digits(0.5)
	selected_X = X_train[np.where((y_train == 4))]
	mixture = BayesMixture(selected_X, K=6, alpha=52, beta=0.5, delta=0.8, seed=7219)
	z_new, a_k, gamma, mixing_props, bad_ak = mixture.gibbs_sample(max_iterations=52, hard_cluster=False)

	images_output = 10
	selected_obs = selected_X[0:images_output]
	predictions = find_clusters(a_k[:, :, -2], selected_obs, partial=True)

	obs_tops = selected_obs[:, 0:392]
	predicted_bottoms = a_k[predictions, 392:, -2]
	stacked = np.hstack([obs_tops, predicted_bottoms])
	rcParams['figure.figsize'] = images_output, 1
	fig, axes = plt.subplots(1, images_output)
	for k in range(0, images_output):
		cax = axes[k % images_output].imshow(stacked[k, :].reshape(28, 28), vmin=0, vmax=1, interpolation="nearest",
											 cmap='plasma')
		axes[k % images_output].set_xlabel("")
		axes[k % images_output].set_ylabel("")
		axes[k % images_output].axis("off")
	plt.tight_layout()
	plt.savefig("fill_in.png", dpi=600)


def fill_in():
	X_train, y_train, X_test, y_test = get_digits(0.5)
	selected_X = X_train[np.where((y_train == 4))]
	mixture = BayesMixture(selected_X, K=10, alpha=52, beta=0.5, delta=0.8, seed=int(time.time()))
	z_new, a_k, gamma, mixing_props, bad_ak = mixture.gibbs_sample(max_iterations=52, hard_cluster=False)

	selected_obs = selected_X[0:10]
	predictions = find_clusters(a_k[:, :, -2], selected_obs, partial=True)

	obs_tops = selected_obs[:, 0:392]
	predicted_bottoms = a_k[predictions, 392:, -2]
	stacked = np.hstack([obs_tops, predicted_bottoms])

	rcParams['figure.figsize'] = 10, 3
	fig, axes = plt.subplots(1, 10)
	for k in range(0, 10):
		cax = axes[k % 10].imshow(stacked[k, :].reshape(28, 28), vmin=0, vmax=1, interpolation="nearest", cmap='plasma')
		axes[k % 10].set_xlabel("")
		axes[k % 10].set_ylabel("")
		axes[k % 10].axis("off")


def plot_intro_clusters():
	X_train, y_train, X_test, y_test = get_digits(0.5)

	selected = X_train[np.where((y_train == 1) | (y_train == 2) | (y_train == 3))]

	rcParams['figure.figsize'] = 3, 3
	mixture = BayesMixture(selected, K=3, alpha=52, beta=0.8, delta=0.8, seed=int(time.time()))
	z_new, a_k, gamma, mixing_props, best_ak = mixture.gibbs_sample(max_iterations=52, hard_cluster=False)
	est_1 = a_k[:, :, -2]

	for i in range(10):
		plt.imshow(selected[i, :].reshape(28, 28), cmap="binary", aspect=1, interpolation='none')
		plt.axis("off")
		plt.tight_layout()
		plt.savefig(f"data_{i}.png", dpi=80)

	for i in range(3):
		plt.imshow(est_1[i, :].reshape(28, 28), cmap="binary", aspect=1, interpolation='none')
		plt.axis("off")
		plt.tight_layout()
		plt.savefig(f"est_{i}.png", dpi=80)


def plot_comparison_histo():
	X_train, y_train, X_test, y_test = get_digits(0.5)
	X_test = X_test[np.where(y_test == 3)]
	with open(f"C:\\data\\bayes_ests_1565890104.431548.bin", "rb") as f:
		bayes_ests = pickle.load(f)
	with open(f"em_ests_3_20.bin", "rb") as f:
		em_ests = pickle.load(f)

	max_ests = 60
	print("Limiting to last", max_ests, "values")

	bayes_best_auroc = []
	em_auroc = []
	bayes_last_auroc = []

	best_ests = [est["best_ak"] for est in bayes_ests]
	averaged_input = chunks(best_ests, 10)
	for idx, chunk in enumerate(averaged_input):
		ref_params = chunk[0]
		aligned_params = [chunk[0]]

		for estimate in chunk[1:]:
			aligned_params.append(estimate[fast_align_labels(ref_params, estimate)[0], :])
		mean_params = np.mean(np.stack(aligned_params), axis=0)

		predictions = find_clusters(mean_params, X_test, partial=True)
		auroc = calculate_auroc(mean_params, X_test, predictions, plot=True)
		bayes_best_auroc.append(auroc)
		print(f"Done chunk {idx}")

	for idx, estimate in enumerate(em_ests[:max_ests]):
		predictions = find_clusters(estimate.T, X_test, partial=True)
		auroc = calculate_auroc(estimate.T, X_test, predictions)
		em_auroc.append(auroc)
		print(f"Done EM {idx}")

	x_grid = np.linspace(0.89, 0.925, 1000)
	plt.ylim((0, 225))

	plt.fill(x_grid, kde_sklearn(em_auroc, x_grid), alpha=0.75, label="EM Model", color="#D95F02")
	plt.fill(x_grid, kde_sklearn(bayes_best_auroc, x_grid), alpha=0.75, label="Bayesian Model", color="#1B9E77")
	plt.xlabel("AUROC")
	plt.ylabel("Probability Density")
	plt.legend(loc="upper left")
	plt.show()


def plot_k_graphs():
	rcParams['figure.figsize'] = 5, 4
	X_train, y_train, X_test, y_test = get_digits(0.5)
	with open(f"ests_k_1565838384.2731788.bin", "rb") as f:
		ests_k = pickle.load(f)

	for digit in ests_k:
		selected_digits = X_test[np.where(y_test == digit)]
		auroc_list = []
		for k in ests_k[digit]:
			estimate = ests_k[digit][k][0]

			predictions = find_clusters(estimate, selected_digits, partial=True)
			auroc_list.append(calculate_auroc(estimate, selected_digits, predictions))
		auroc_list = 100 - (np.array(auroc_list) / auroc_list[0] * 100)
		fig, ax = plt.subplots()

		ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
		plt.axhline(y=0, alpha=0.5, c="black")
		plt.plot(list(ests_k[digit].keys()), auroc_list, label=f"{digit}")
		print(f"Done {digit}")
		plt.xlabel("K")
		plt.ylim((-1, 1))
		plt.ylabel("Relative increase in AUROC")
		plt.title(f"Digit {digit}")
		plt.tight_layout()
		plt.savefig(f"d_{digit}_kplots.png", dpi=100)
		plt.close()


def plot_digits_switching():
	X_train, y_train, X_test, y_test = get_digits(0.5)

	for digit in [1, 2, 3]:
		selected = X_train[np.where((y_train == digit))]

		mixture = BayesMixture(selected, K=3, alpha=52, beta=0.5, delta=0.8, seed=int(time.time()))
		z_new, a_k, gamma, mixing_props = mixture.gibbs_sample(max_iterations=52, hard_cluster=False)
		est_1 = a_k[:, :, -1]

		for i in range(3):
			plt.imshow(est_1[i, :].reshape(28, 28), cmap="plasma", aspect=1, interpolation='none', vmin=0, vmax=1)
			plt.axis("off")
			plt.tight_layout()
			plt.savefig(f"d{digit}-{i}.png", )


if __name__ == "__main__":
	# plot_clusters()
	# plot_intro_clusters()
	plot_comparison_histo()
