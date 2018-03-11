import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.decomposition import TruncatedSVD, NMF


def load_data():
	data =  pd.read_csv('data/ratings.csv')
	return data

def plot_mae_n_rmse(mae_errors, rmse_error, ks, min_k, filename):
	fig = plt.figure(figsize=(12, 6))
	errors = [mae_errors, rmse_error]
	labels = ['mean absolute error', 'root mean square error']
	for i, errors in enumerate(errors):

		ax = fig.add_subplot(1, 2, i+1)
		ax.scatter(ks, errors)
		ax.set_xlabel('k')
		ax.set_ylabel(labels[i])

		if min_k is not None:
			min_k_mark = ax.scatter(ks[min_k], errors[min_k], c='red', marker='*')
			ax.legend([min_k_mark], ['minimum_k'])

		ax.grid('on')

	fig.tight_layout()
	fig.savefig('report/figures/'+filename, dpi=300)
