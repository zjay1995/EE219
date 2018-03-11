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


def count_like_items(series):
    return np.sum(np.where(series.values>3, 1, 0))


def build_tst(tst, prediction):
    temp = pd.DataFrame({
    'userId': [int(pred[0]) for pred in prediction],
    'movieId': [int(pred[1]) for pred in prediction],
    'y_pred': [pred[3] for pred in prediction]
    })
    df = pd.merge(tst, temp, on=['movieId', 'userId'], how='left').\
        rename(columns={'rating': 'y_true'})
    df = remove_zero_likes_users(df)
    return df.groupby('userId')


def remove_zero_likes_users(df):
    temp = df.groupby('userId')
    temp = temp.agg({'y_true': count_like_items})
    temp = temp[temp.y_true == 0].index
    df = df[~df.userId.isin(temp)]
    return df


def compute_ppv_tpr(df, *args):
    ts = args[:-1]
    threshold = args[-1]
    rank = df.sort_values('y_pred', ascending=False).movieId.values.tolist()
    like = set(df[df.y_true > threshold].movieId.values)
    ppv = [np.nan] * len(ts)
    tpr = [np.nan] * len(ts)
    for i, t in enumerate(ts[:np.min([len(ts), len(rank)])]):
        overlap = len(like.intersection(rank[:t]))
        ppv[i] = overlap / len(rank[:t])
        tpr[i] = overlap / len(like)
    return np.array(ppv + tpr)