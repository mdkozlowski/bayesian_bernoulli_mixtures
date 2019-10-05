# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'

from multiprocessing.pool import Pool
import pickle
import time

import h5py
import numpy as np
from .util import get_digits
from .gibbs import BayesMixture, BernoulliMixtureEM
from .metrics import average_of_estimates, find_clusters, calculate_auroc


def threaded_estimates(data):
	X_train = data[0]
	K = data[1]
	start_time = time.time()
	mixture = BayesMixture(X_train, K=K, alpha=50, beta=0.5, delta=0.8, seed=int(time.time()))
	z_new, a_k, gamma, mixing_props = mixture.gibbs_sample(max_iterations=30, hard_cluster=False, verbose=False)
	duration = time.time() - start_time
	return a_k[:, :, -1], duration


def threaded_estimates_EM(data):
	X_train = data[0]
	K = data[1]
	start_time = time.time()
	mixture = BernoulliMixtureEM(K=K, D=784, X=X_train)
	a_k, pi = mixture.train(max_iterations=80, verbose=False, display_interval=50, save_interval=None)
	duration = time.time() - start_time
	return a_k[:, :, -1], duration


def get_data(X_train, n=1100, K=8):
	for i in range(n):
		yield (X_train, K)


def collect_estimates():
	X_train, y_train, X_test, y_test = get_digits(0.5)

	cores = 6
	pool = Pool(processes=cores)
	for k in [10]:
		for digit_estimated in [3, 4, 5, 6, 7, 8, 9]:
			condition = (y_train == digit_estimated)
			selected_digits = X_train[np.where(condition)]
			selected_labels = y_train[np.where(condition)]

			h5 = h5py.File(f"./datasets/{digit_estimated}_estimates_{k}.h5", "a")
			remaining_digits = 250 - len(h5.keys())
			durations = []
			idx = 0
			for estimate, duration in pool.imap_unordered(threaded_estimates,
														  get_data(selected_digits, n=remaining_digits, K=k)):
				h5.create_dataset(f"{idx}{time.time()}", data=estimate)
				durations.append(duration)
				if idx % 20 == 0:
					print(
						f"{digit_estimated} - Done with {idx}/{remaining_digits}. Avg {np.mean(durations) / cores:.3f}s per estimate.")
				idx += 1
			h5.close()


def collect_estimates_histo():
	X_train, y_train, X_test, y_test = get_digits(0.5)
	bayes_ests = []
	em_ests = []
	for i in range(1000):
		condition = (y_train == 3)
		selected_digits = X_train[np.where(condition)]

		mixture = BayesMixture(selected_digits, K=20, alpha=50, beta=0.5, delta=0.8, seed=int(time.time()))
		z_new, a_k, gamma, mixing_props, best_ak = mixture.gibbs_sample(max_iterations=50, hard_cluster=False,
																		verbose=False)
		bayes_ests.append({"best_ak": best_ak,
						   "all_ak": a_k,
						   "all_z": z_new})
		if i % 10 == 0:
			print(f"Done {i} out of 1000 for Bayes")
		if i % 100 == 0:
			with open(f"D:\\bayes_ests_{time.time()}.bin", "wb") as f:
				pickle.dump(bayes_ests, f)
	exit()

	print("Done histo")


def collect_estimates_k():
	X_train, y_train, X_test, y_test = get_digits(0.5)

	bayes_ests = {}
	with open(f"ests_k_1565828825.666521.bin", "rb") as f:
		bayes_ests = pickle.load(f)
	for digit in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
		print(f"Doing digit {digit}")
		if digit not in bayes_ests:
			bayes_ests[digit] = {}

		for k in [10, 20, 30, 50]:
			print(f"Doing k= {k}")
			if k in bayes_ests[digit]:
				continue

			condition = (y_train == digit)
			selected_digits = X_train[np.where(condition)]
			local_est = []

			for est in range(10):
				mixture = BayesMixture(selected_digits, K=20, alpha=50, beta=0.5, delta=0.8, seed=int(time.time()))
				z_new, a_k, gamma, mixing_props = mixture.gibbs_sample(max_iterations=50, hard_cluster=False,
																	   verbose=False)
				local_est.append(a_k[:, :, -1])

			bayes_ests[digit][k] = average_of_estimates(local_est, 10)

			with open(f"ests_k_{time.time()}.bin", "wb") as f:
				pickle.dump(bayes_ests, f)
			print(f"Done digit {digit}, K={k}")


def collect_estimates_EM():
	X_train, y_train, X_test, y_test = get_digits(0.5)
	h5 = h5py.File(f"./datasets/EM_estimates_{10}.h5", "a")

	for digit in [0, 1, 2, 3]:
		condition = (y_train == digit)
		selected_digits = X_train[np.where(condition)]

		mixture = BernoulliMixtureEM(K=10, D=784, X=selected_digits)
		a_k, pi = mixture.train(max_iterations=100, display_interval=20, save_interval=None, verbose=False)
		h5.create_dataset(f"{digit}", data=a_k.T)
	h5.close()


def save_auroc(mode="bayes"):
	X_train, y_train, X_test, y_test = get_digits(0.5)

	cores = 4
	pool = Pool(processes=cores)
	estimates = {}
	for k in [20, 30]:
		estimates[k] = {}
		for digit_estimated in [0, 1, 2, 3]:
			condition = (y_train == digit_estimated)
			selected_digits = X_train[np.where(condition)]

			durations = []
			local_estimates = []
			if mode == "bayes":
				est_fun = threaded_estimates
			else:
				est_fun = threaded_estimates_EM

			for estimate, duration in pool.imap_unordered(est_fun, get_data(selected_digits, n=10, K=k)):
				local_estimates.append(estimate)

			estimates[k][digit_estimated] = average_of_estimates(local_estimates, 10)
			print(f"Done k={k}, digit #{digit_estimated}")
	with open(f"{mode}_auroc_allk.bin", "wb") as f:
		pickle.dump(estimates, f)


def simple_compare():
	X_train, y_train, X_test, y_test = get_digits(0.5)
	with open(f"bayes_auroc_allk.bin", "rb") as f:
		bayes_data = pickle.load(f)

	for k in [20, 30]:
		for digit_estimated in [0, 1, 2, 3]:
			avg_est = bayes_data[k][digit_estimated][0]
			predictions = find_clusters(avg_est, X_test, partial=True)
			auroc = calculate_auroc(avg_est, X_test, predictions)
			print(f"#{digit_estimated}:K={k}: {auroc:.4f}")


def final_compare():
	X_train, y_train, X_test, y_test = get_digits(0.5)

	with h5py.File(f"./datasets/EM_estimates_10.h5", "r") as h5:
		bernoulli_estimates = {key: h5[key][()] for key in h5.keys()}

	for digit in [0, 1, 2, 3]:
		print(f"Doing digit {digit}")

		with h5py.File(f"./datasets/{digit}_estimates_10.h5", "r") as h5:
			bayes_estimates = [h5[key][()] for key in h5.keys()]

		avg_bayes_set = average_of_estimates(bayes_estimates, size=10)[0:10]

		bayes_auroc = []
		for idx, estimate in enumerate(avg_bayes_set):
			predictions = find_clusters(estimate, X_test, partial=True)
			auroc = calculate_auroc(estimate, X_test, predictions)
			bayes_auroc.append(auroc)
		print("Done calculating Bayes AUROC")

		predictions = find_clusters(bernoulli_estimates[f"{digit}"].T, X_test, partial=True)
		auroc = calculate_auroc(bernoulli_estimates[f"{digit}"].T, X_test, predictions)

		print(f"Bayes best estimate: {np.min(bayes_auroc):.4f}")
		print(f"EM estimate: {auroc:.4f}")


if __name__ == "__main__":
	collect_estimates_histo()
