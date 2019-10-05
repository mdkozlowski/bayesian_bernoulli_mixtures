import gzip
import pickle

import numpy as np
from scipy.stats import bernoulli
from sklearn.neighbors import KernelDensity


def get_digits(threshold=0.8):
	path = 'C:\\data\\mnist.pkl.gz'

	with gzip.open(path, 'rb') as f:
		train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

		training_digits = np.vstack([train_set[0], valid_set[0]])
		training_digits = np.where(training_digits > threshold, 1, 0)
		training_labels = np.hstack([train_set[1], valid_set[1]])

		test_digits = test_set[0]
		test_digits = np.where(test_digits > threshold, 1, 0)
		test_labels = test_set[1]

	return training_digits, training_labels, test_digits, test_labels


def generate_data(K=3, D=6, cluster_size=500):
	N_each = np.array(np.repeat(cluster_size, K))
	p = np.random.dirichlet(np.full(K, 1), size=D).T

	X = np.hstack([[bernoulli.rvs(j, size=N_each[i]) for j in p[i]] for i in np.arange(0, 3)]).T

	return X, p


def chunks(l, n):
	n = max(1, n)
	return (l[i:i + n] for i in range(0, len(l), n))


def kde_sklearn(x, x_grid, bandwidth=0.0007, **kwargs):
	kde_skl = KernelDensity(bandwidth=bandwidth, kernel="gaussian", **kwargs)
	kde_skl.fit(np.array(x)[:, np.newaxis])
	log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
	return np.exp(log_pdf)
