import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import chain, combinations
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def cross_validate_custom(nfolds, seed, x, y, clf):
    '''
        Customize cross validation routine,
        with access to y_pred of each fold.
    '''
    y_pred = np.zeros(y.shape)
    trn_rmse = np.zeros((nfolds, ))
    folds = KFold(n_splits=nfolds, shuffle=True, random_state=seed).split(y)

    for i, (trn_idx, tst_idx) in enumerate(folds):
        x_trn = x[trn_idx, :]
        x_tst = x[tst_idx, :]
        y_trn = y[trn_idx]  
        
        clf.fit(x_trn, y_trn)
        y_pred[tst_idx] = clf.predict(x_tst)
        trn_rmse = np.sqrt(mean_squared_error(y_trn, clf.predict(x_trn)))
    
    return y_pred, np.mean(trn_rmse)


def evaluate_results(trn_rmse, tst_rmse, y_true, y_pred, filename):
    print("train rmse: %f" % trn_rmse)
    print("test rmse: %f" % tst_rmse)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax1.scatter(y_pred, y_true)
    ax1.set_xlabel("Prediction")
    ax1.set_ylabel("True")

    ax2 = fig.add_subplot(122)
    ax2.scatter(y_pred, y_pred-y_true)
    ax2.set_xlabel("Prediction")
    ax2.set_ylabel("Residual")

    fig.tight_layout()
    fig.savefig('report/figures/'+filename, dpi=300)


def powerset(iterable):
    '''
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
        return generator instead of list iterable.
    '''
    xs = list(iterable)
    return chain.from_iterable(combinations(xs, n) for n in range(len(xs)+1)) 


def one_hot_encode(df, cols):
    '''
        By default drop the original label-encode feature    
    '''
    df_temp = df.copy()
    for col in cols:
        temp = pd.get_dummies(df_temp[col], prefix=col)
        df_temp = pd.concat([df_temp, temp], axis=1)

    df_temp = df_temp.drop(cols, axis=1)
    return df_temp

def find_best_ohe_subsets(nfolds, seed, x, y, features, clf):
    '''
        For finding the best ohe subsets.
    '''
    trn_rmses = np.zeros((2**5, ))
    tst_rmses = np.zeros((2**5, ))

    feature_sub_collector = []

    for i, feature_sub in enumerate(powerset(features)):
        feature_sub_collector.append(list(feature_sub))
        x_temp = one_hot_encode(x, list(feature_sub))
        
        y_pred, trn_rmse = cross_validate_custom(nfolds, seed, x_temp.values, y, clf)
        tst_rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        trn_rmses[i] = trn_rmse
        tst_rmses[i] = tst_rmse

    return trn_rmses, tst_rmses, feature_sub_collector

def plot_best_ohe_subsets(trn_rmses, tst_rmses, feature_sub_collector, filename):
    
    print('ohe that produces best train: ', feature_sub_collector[np.argmin(trn_rmses)])
    print('ohe that produces best test: ', feature_sub_collector[np.argmin(tst_rmses)])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(trn_rmses, '--')
    ax.plot(tst_rmses, '-')
    ax.scatter(np.argmin(trn_rmses), np.min(trn_rmses), marker='o', color='g')
    ax.scatter(np.argmin(tst_rmses), np.min(tst_rmses), marker='o', color='r')
    ax.legend(['train rmse', 'test rmse', 'min. train rmse', 'min. test rmse'])
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Root mean square error')

    fig.tight_layout()
    fig.savefig('report/figures/'+filename, dpi=300)