import itertools
import warnings

import numpy as np
from scipy.stats import norm, bernoulli
import pandas as pd
import time as time

from .metrics import find_clusters
import pickle
import multiprocessing as mp


class BernoulliMixtureEM:
	def __init__(self, K=3, D=4, X=np.zeros((100, 4))):
		self.K = K
		self.D = D
		self.X = X
		self.N = self.X.shape[0]

		self.z = np.zeros((self.N, self.K))
		self.mu = (np.random.rand(self.D, self.K) * 0.5) + 0.25
		self.mu /= self.mu.sum(axis=0)
		self.pi = np.full((self.K), 1 / self.K)
		print(f"Done initialising model with K={self.K}, N={self.N}, D={self.D}")

	def expectation(self):
		for n in range(self.N):
			x_n = self.X[n, :]
			x_tiled = np.tile(x_n, (self.K, 1)).T

			z_n = np.prod(np.power(self.mu, x_tiled) * np.power(1 - self.mu, 1 - x_tiled), axis=0) * self.pi
			summation = z_n.sum()
			if summation > 0:
				self.z[n, :] = z_n / summation
			else:
				self.z[n, :] = np.full(self.K, 1 / self.K)

	def maximisation(self):
		self.pi = np.sum(self.z, axis=0) / self.K
		self.mu = np.dot(self.X.T, self.z) / np.sum(self.z, axis=0)

	def train(self, max_iterations=100, display_interval=100, save_interval=5, verbose=False):
		for idx in range(max_iterations):
			iter_start = time.time()
			self.expectation()
			iter_exp = time.time()
			self.maximisation()
			self.pi /= self.pi.sum()
			iter_end = time.time()
			if idx % display_interval == 0 and verbose:
				print(
					f"Iteration {idx} out of {max_iterations}. E: {iter_exp - iter_start: 3f}, M: {iter_end - iter_exp: 3f}, T: {iter_end - iter_start: 3f}")
			if save_interval is not None and idx % save_interval == 0:
				np.save(f'mu-{time.time():.3f}', self.mu)
				np.save(f'pi-{time.time():.3f}', self.pi)
		return self.mu, self.pi

	def print_estimates(self):
		print(np.round(self.mu.T, 3))
		print(np.round(self.pi, 3))


class BayesMixture:
	def __init__(self, data, K, alpha, beta, delta, seed):
		self.X = data
		self.D = data.shape[1]
		self.N = data.shape[0]
		self.K = K
		self.alpha = alpha
		self.beta = beta
		self.delta = delta

		self.random_state = np.random.RandomState(seed)
		self.z = np.zeros((self.N, self.K))
		self.a_k = None

	def gibbs_sample(self, max_iterations=3, epsilon=1.0e-6, minimum_iterations=10, sweeps=100, hard_cluster=False,
					 verbose=True, early_stopping=True):
		self.z = np.zeros((self.N, self.K))

		temp_z = np.tile(np.eye(self.K), (int(np.ceil(self.N / self.K)), 1))
		np.random.shuffle(temp_z)
		self.z = temp_z[:self.N, :self.K]

		z_new = np.zeros((self.N, self.K, max_iterations))
		z_new[:, :, 0] = self.z

		gamma = np.zeros((self.N, self.K, max_iterations))
		a_k = np.zeros((self.K, self.D, max_iterations))
		mixing_props = np.zeros((self.K, max_iterations))
		iteration_time = []
		inners = []

		for idx in range(1, max_iterations):
			if verbose:
				print(f"iteration {idx}")
			start = time.time()

			x_in_k = [self.X[np.where(z_new[:, k, idx - 1])] for k in range(self.K)]
			sum_xd_in_k = np.array([np.sum(x, axis=0) for x in x_in_k])
			N_k = np.array([x.shape[0] for x in x_in_k])
			N_notnk = N_k - z_new[:, :, idx - 1]
			coeff = np.log((N_notnk + (self.alpha / self.K)) / (self.N - 1 + self.alpha))
			prod_den = np.log(self.beta + self.delta + N_k) * self.D
			start_inner = time.time()
			mixing_props[:, idx] = N_k / self.N
			for n in range(self.N):
				left = np.power(self.beta + sum_xd_in_k[:, :], self.X[n, :])
				right = np.power(self.delta + N_k.repeat((self.D)).reshape(self.K, self.D) - sum_xd_in_k,
								 1 - self.X[n, :])
				prod_num = np.sum(np.log(left * right), axis=1)
				val = np.exp(coeff[n, :] + prod_num - prod_den)
				gamma[n, :, idx] = val / sum(val)

			end_inner = time.time()
			inners.append(end_inner - start_inner)

			if hard_cluster:
				choices = gamma[:, :, idx].argmax(axis=1)
				for n in range(self.N):
					z_new[n, choices[n], idx] = 1
			else:
				cumulative_probs = np.cumsum(gamma[:, :, idx], axis=1)
				u = np.random.rand(len(cumulative_probs), 1)
				choices = (u < cumulative_probs).argmax(axis=1)
				for n in range(self.N):
					z_new[n, choices[n], idx] = 1

			for k in range(self.K):
				x_in_k = self.X[np.where(z_new[:, k, idx] == 1)]
				with warnings.catch_warnings():
					warnings.filterwarnings('error')
					if x_in_k.shape[0] == 0:
						a_k[k, :, idx] = 0
					else:
						try:
							a_k[k, :, idx] = np.sum(x_in_k, axis=0) / x_in_k.shape[0]
						except Warning as e:
							# pass
							print('error found:', e)

			delta_assignments = (np.mean(np.abs(np.diff(gamma, axis=2, n=2)), axis=(0, 1)))[idx - 2]
			if 0 < delta_assignments < 0.002 and early_stopping:
				a_k = np.delete(a_k, axis=2, obj=np.arange(idx + 2, max_iterations))
				z_new = np.delete(z_new, axis=2, obj=np.arange(idx + 2, max_iterations))
				gamma = np.delete(gamma, axis=2, obj=np.arange(idx + 2, max_iterations))
				mixing_props = np.delete(mixing_props, axis=1, obj=np.arange(idx + 2, max_iterations))
				end = time.time()
				iteration_time.append(end - start)
				break

			end = time.time()
			iteration_time.append(end - start)

		if verbose:
			print(
				f"{np.mean(np.array(iteration_time) * 1.0e8) / (self.K * self.N * self.D):.3f} ops per iteration with {idx - 1} total iterations")
			print(
				f"{np.mean(np.array(inners)) * 1000:.2f}ms inner loop, {np.mean(np.array(inners) / np.array(iteration_time)) * 100:.2f}% of my time")
		self.a_k = a_k
		self.best_ak = np.mean(a_k[:, :, -5:], axis=2)
		return z_new, a_k, gamma, mixing_props, self.best_ak

	def find_clusters(self, data_obs, alignments=None, partial=False):
		return find_clusters(self.best_ak, data_obs, alignments, partial)

	def load(self, filename):
		f = open(filename, 'rb')
		tmp_dict = pickle.load(f)
		f.close()

		self.__dict__.update(tmp_dict)

	def save(self, filename):
		f = open(filename, 'wb')
		pickle.dump(self.__dict__, f, 2)
		f.close()
