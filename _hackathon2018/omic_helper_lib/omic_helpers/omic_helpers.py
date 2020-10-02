import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import scipy as sc
from scipy.interpolate import PchipInterpolator as minterp
import scipy.stats as stats
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

import os
import sys
import re
from numba import jit
from collections import Counter
from collections import namedtuple
from collections import defaultdict

from sklearn.covariance import OAS
from sklearn.covariance import GraphicalLassoCV
from sklearn.covariance import LedoitWolf
from sklearn.covariance import EmpiricalCovariance

from scipy.stats import ks_2samp as ks, wasserstein_distance as wass, spearmanr, energy_distance, pearsonr
from scipy.stats import chisquare, epps_singleton_2samp as epps

from sklearn.decomposition import PCA, FastICA as ICA, FactorAnalysis as FA, MiniBatchSparsePCA as SparsePCA
from sklearn.decomposition import MiniBatchDictionaryLearning as DictLearn, NMF
from sklearn.feature_selection import mutual_info_classif as Minfo, f_classif as Fval, chi2

import logging

'''

TO-DO's :

class binomial_feature_evaluator

class multinomial_feature_evaluator

class regression_feature_evaluator

class ts_feature_evaluator 

class feature_expansion
    bulk_feature_expander
    iterative_feature_expander

separate functions:
recursive_feature_splitter
information_gain
association_rule_miner
distance correlation/covariance
maximal correlation
Bayes factor
Random Matrix Theoretical signal
..
..

Specifically:
    Add types to @jit decorators
    Remove if-statements from @jit functions
'''


def get_statdist_dataframe_binomial(X,Y, features):
    assert features is not None, 'You are required to provide the feature list,\n\r\t\t otherwise the dataframe cannot be indexed'
    assert len(features)== X.shape[1], 'The number of feature names given is not equal to the number of columns'
    assert len(Y) == X.shape[0], 'The number of rows in the matrix X is not the equal to the length of the target vector Y'
    assert X.isna().sum().sum()==0, 'There are NaN\'s present in this dataset, please remove them' 

    if features is None:
        features = ['feat_' + i for i in range(0, X.shape[1])]

    num_features = len(features)
    nanvector = np.empty((num_features,))
    nanvector[:] = np.nan

    stat_dist = dict()
    logging.info("Processing diffentropy, variance scores, spca/pca/fa/ica/nmf/dl importances...")
    logging.info("diff entropy..")
    stat_dist['diffentropy'] = diff_entropy_scores(X, eps=1e-6, bins=30)
    logging.info("variance..")
    stat_dist['variance'] = variance_scores(X)
    logging.info("pca_imp..")
    stat_dist['pca_imp'] = get_reducer_weights(X, PCA(n_components=num_features), cols=None, ncomp=num_features, weighted='expl_variance')
    logging.info("spca_imp..")
    stat_dist['spca_imp'] = get_reducer_weights(X, SparsePCA(n_components=num_features, batch_size=5, method='lars'), cols=None, ncomp=num_features, weighted='linear')
    logging.info("fa_imp..")
    stat_dist['fa_imp'] = get_reducer_weights(X, FA(n_components=num_features), cols=None, ncomp=num_features, weighted='noise_variance')
    logging.info("ica_imp..")
    stat_dist['ica_imp'] = get_reducer_weights(X, ICA(n_components=num_features), cols=None, ncomp=num_features, weighted=None)
    logging.info("nmf_imp..")
    stat_dist['nmf_imp'] = get_reducer_weights(X, NMF(n_components=num_features), cols=None, ncomp=num_features, weighted='linear')
    logging.info("dl_imp..")
    stat_dist['dl_imp'] = get_reducer_weights(X, DictLearn(n_components=num_features), cols=None, ncomp=num_features, weighted=None)

    print("Processing minf/fscore/wass1/wass2...")
    logging.info("minf..")
    stat_dist['minf'] = Minfo(X, Y)
    logging.info("fscore..")
    stat_dist['fscore'] = Fval(X, Y)

    logging.info("Wass1..")
    fswass1 = fs_ws1()
    stat_dist['Wass1'] = fswass1.fit(X.values, Y).scores_

    fswass2 = fs_ws2()
    logging.info("Wass2..")
    stat_dist['Wass2'] = fswass2.fit(X.values, Y).scores_

    print("Processing spearman and ks...")
    logging.info("spearman..")
    stat_dist['spearman'] = spearman_scores(X, Y)
    logging.info("ks..")
    stat_dist['ks'] = ks_scores(X, Y)

    print("Processing seq entropies...")
    try:
        logging.info("seqentropy..")
        stat_dist['seqentropy'] = seq_entropy_scores(X, Y)
        logging.info("qseqentropy_prod..")
        stat_dist['qseqentropy_prod'] = qseq_entropy_scores(X, Y, q_type='prod', bins=10)
        logging.info("qseqentropy_sum..")
        stat_dist['qseqentropy_sum'] = qseq_entropy_scores(X, Y, q_type='sum', bins=10)
        logging.info("seqentropyX..")
        stat_dist['seqentropyX'] = seq_entropyX_scores(X, Y)
    except Exception as e:
        print("Sequential entropy scores failed: {}".format(e))
        logging.info("seqentropy..")
        stat_dist['seqentropy'] = np.empty((num_features,)).fill(np.nan)
        logging.info("qseqentropy_prod..")
        stat_dist['qseqentropy_prod'] = np.empty((num_features,)).fill(np.nan)
        logging.info("qseqentropy_sum..")
        stat_dist['qseqentropy_sum'] = np.empty((num_features,)).fill(np.nan)
        logging.info("seqentropyX..")
        stat_dist['seqentropyX'] = np.empty((num_features,)).fill(np.nan)

    print("Processing CDF scores...")
    try:
        logging.info("cdf_1..")
        stat_dist['cdf_1'] = cdf_scoresB(X,Y, dist_type='mink_rao')
        logging.info("cdf_2..")
        stat_dist['cdf_2'] = cdf_scoresB(X,Y, dist_type='mink_rao2')
        logging.info("cdf_3..")
        stat_dist['cdf_3'] = cdf_scoresB(X,Y, dist_type='rao')
        logging.info("cdf_4..")
        stat_dist['cdf_4'] = cdf_scoresG(X,Y, dist_type='emd')
        logging.info("cdf_5..")
        stat_dist['cdf_5'] = cdf_scoresG(X,Y, dist_type='cvm')
        logging.info("cdf_6..")
        stat_dist['cdf_6'] = cdf_scoresG(X,Y, dist_type='cust')
    except Exception as e:
        print("CDF scores failed: {}".format(e))
        stat_dist['cdf_1'] = nanvector
        stat_dist['cdf_2'] = nanvector
        stat_dist['cdf_3'] = nanvector
        stat_dist['cdf_4'] = nanvector
        stat_dist['cdf_5'] = nanvector
        stat_dist['cdf_6'] = nanvector

    print("Processing q distances...")
    try:
        logging.info("med_dist..")
        stat_dist['med_dist'] = q_dists(X, Y, q=0.5)
        logging.info("q25_dist..")
        stat_dist['q25_dist'] = q_dists(X, Y, q=0.25)
        logging.info("q75_dist..")
        stat_dist['q75_dist'] = q_dists(X, Y, q=0.75)
        logging.info("var_dist..")
        stat_dist['var_dist'] = var_dists(X, Y)
        logging.info("q5_acc..")
        stat_dist['q5_acc'] = q_acc_scores(X, Y, q=0.5)
        logging.info("q75_acc..")
        stat_dist['q75_acc'] = q_acc_scores(X, Y, q=0.75)
    except Exception as e:
        print("Q distance scores failed: {}".format(e))
        stat_dist['med_dist'] = nanvector
        stat_dist['q25_dist'] = nanvector
        stat_dist['q75_dist'] = nanvector
        stat_dist['var_dist'] = nanvector
        stat_dist['q5_acc'] = nanvector
        stat_dist['q75_acc'] = nanvector

    print("Getting conditional probability of exceedence")
    try:
        logging.info("prob_exc..")
        stat_dist['prob_exc'] = prob_exceed_scores(X, Y)
    except Exception as e:
        print("Getting exceedence proba failed...{}".format(e))
        stat_dist['prob_exc'] = nanvector

    print("Processing cross entropies over class-seperated features...")
    try:
        logging.info("KL..")
        stat_dist['KL'] = ec_scores2(X, Y, num_bins=11, ent_type='kl')
        logging.info("Shan..")
        stat_dist['Shan'] = ec_scores2(X, Y, num_bins=11, ent_type='shannon')
        logging.info("Cross..")
        stat_dist['Cross'] = ec_scores2(X, Y, num_bins=11, ent_type='cross')
    except Exception as e:
        print("Cross entropies failed: {}".format(e))
        stat_dist['KL'] = nanvector
        stat_dist['Shan'] = nanvector
        stat_dist['Cross'] = nanvector

    print("Processing cross entropies over random selects of feature-sorted and non-feature-sorted target vectors")
    try:
        logging.info("KL_sort..")
        stat_dist['KL_sort'] = ecs_scores(X, Y, num_bins=21, ent_type='kl')
        logging.info("Shan_sort..")
        stat_dist['Shan_sort'] = ecs_scores(X, Y, num_bins=21, ent_type='shannon')
        logging.info("Cross_sort..")
        stat_dist['Cross_sort'] = ecs_scores(X, Y, num_bins=21, ent_type='cross')
    except Exception as e:
        print("Cross entropies failed: {}".format(e))
        stat_dist['KL_sort'] = nanvector
        stat_dist['Shan_sort'] = nanvector
        stat_dist['Cross_sort'] = nanvector

    print("Processing Chi2 and Epps...")
    try:
        fsepps = fs_epps(pvalue=0.01)
        logging.info("Chi2..")
        stat_dist['Chi2'] = chi2_scores(X.values, Y, bins=7)
        logging.info("epps..")
        stat_dist['epps'] = fsepps.fit(X.values, Y).scores_
    except Exception as e:
        print("Cross Chi2/Epps failed: {}".format(e))
        stat_dist['Chi2'] = nanvector
        stat_dist['epps'] = nanvector


    # Combine in dataframe
    stat_dist_df = pd.DataFrame(data=np.vstack([stat_dist['diffentropy'],
                                                stat_dist['variance'],
                                                stat_dist['pca_imp'],
                                                stat_dist['spca_imp'],
                                                stat_dist['fa_imp'],
                                                stat_dist['ica_imp'],
                                                stat_dist['nmf_imp'],
                                                stat_dist['dl_imp'],
                                                stat_dist['minf'], 
                                                stat_dist['fscore'], 
                                                stat_dist['Wass1'],
                                                stat_dist['Wass2'],
                                                stat_dist['spearman'].T, 
                                                stat_dist['ks'].T,
                                                stat_dist['KL'],
                                                stat_dist['Shan'],
                                                stat_dist['Cross'],
                                                stat_dist['KL_sort'],
                                                stat_dist['Shan_sort'],
                                                stat_dist['Cross_sort'],
                                                stat_dist['seqentropy']*stat_dist['Wass1'],
                                                stat_dist['qseqentropy_prod']*stat_dist['Wass1'],
                                                stat_dist['qseqentropy_sum']*stat_dist['Wass1'],
                                                stat_dist['seqentropyX']*stat_dist['Wass1'],
                                                stat_dist['cdf_1'],
                                                stat_dist['cdf_2'],
                                                stat_dist['cdf_3'],
                                                stat_dist['cdf_4'],
                                                stat_dist['cdf_5'],
                                                stat_dist['cdf_6'],
                                                stat_dist['med_dist'],
                                                stat_dist['q25_dist'],
                                                stat_dist['q75_dist'],
                                                stat_dist['var_dist'],
                                                stat_dist['q5_acc'], 
                                                stat_dist['q75_acc'],
                                                stat_dist['prob_exc'],
                                                stat_dist['prob_exc']*stat_dist['Wass1'],
                                                stat_dist['Chi2'],
                                                stat_dist['epps']]).T,
                               columns=['diffentropy', 'variance',
                                        'pca_imp', 'spca_imp', 'fa_imp', 'ica_imp', 'nmf_imp', 'dl_imp',
                                        'Minf', 
                                        'Fscore', 'FPval',
                                        'Wass1', 'Wass2',
                                        'SpearmanScore', 'SpearmanPval',
                                        'KSScore', 'KSPval', 'KL', 'Shannon', 'Cross', 'KL_sort', 'Shannon_sort', 'Cross_sort',
                                        'seqentropy_wass1', 'qseqentropy_prod_wass1', 'qseqentropy_sum_wass1', 'seqentropyX_wass1',
                                        'CDF1', 'CDF2', 'CDF3', 'CDF4', 'CDF5', 'CDF6',
                                        'q5delta', 'q25delta', 'q75delta', 'var_dist', 'q5_acc', 'q75_acc', 'prob_exc',
                                        'prob_exc_wass1', 'Chi2', 'Epps'],
                               index=features)
    return stat_dist_df    

def _feature_selector(dists_df, topN=100, overlap='intersect',
                      score_list=['qseqentropy_sum_wass1', 'spca_imp', 'Chi2',
                                  'q5delta', 'Wass1', 'Minf', 'CDF6', 'prob_exc_wass1']):
    '''
    :param dists_df: dataframe with features and univariate scores
    :param topN: top N rank of features
    :param overlap: take the 'intersect' of the top N features,  or the 'union'
    :return:
    '''
    if score_list == None:
        score_list = dists_df.columns.tolist()
    _features = dists_df.index.tolist()
    ignore = 'Pval'
    cols = [_col for _col in score_list if ignore not in _col]
    dists_df = dists_df.loc[:, cols].abs()

    totset = set()
    for _col in dists_df.columns:
        add = set(dists_df.sort_values(by=_col, ascending=False)[:topN].index)
        if overlap=='union':
            totset = totset.union(add)
        else:
            totset = totset.intersection(add)
    return list(totset)
   
def get_reducer_weights(X, reducer=None, cols=None, ncomp=10, weighted=None):
    '''
     reducer: e.g PCA, FA, ICA, DL, NMF
    '''
    reducer.fit(X)    
    pcweights = np.zeros((X.shape[1],))
    wt = 0
    for pc in range(0, ncomp):
        if weighted=='linear':
            w = ncomp-pc
            wt = wt + w
            pcweights = w*np.abs(reducer.components_[pc]) + pcweights
        elif weighted=='expl_variance':
            w = reducer.explained_variance_ratio_[pc]
            wt = wt + w
            pcweights = w*np.abs(reducer.components_[pc]) + pcweights
        elif weighted=='noise_variance':
            w = reducer.noise_variance_[pc]
            wt = wt + w
            pcweights = w*np.abs(reducer.components_[pc]) + pcweights
        else:
            wt = wt + 1
            pcweights = np.abs(reducer.components_[pc]) + pcweights
    pcweights = pcweights/wt    
    if cols is not None:
        pcw = pcweights[np.argsort(pcweights)]
        _cols = np.array(cols)[np.argsort(pcweights)]
        return dict(zip(_cols,pcw))
    else:
        return pcweights



def featurewise_outlier_replacer(X, q=(0.01, 0.99)):
    if "DataFrame" in str(type(X)):
        X = X.values
    Xnew = np.zeros((X.shape[0], X.shape[1]))
    for jdx in range(0, X.shape[1]):
        x = X[:, jdx]
        Xnew[:, jdx] = _featurewise_outlier_replacer(x, q)
    return Xnew

@jit
def _featurewise_outlier_replacer(x, q=(0.01, 0.99), how='quantile'):
    if how == 'quantile':
        # improvement..instead of blunt replacement with fixed number use sampler
        lv, rv = np.quantile(x, q[0]), np.quantile(x, q[1])
        t = x.copy()
        t[t > rv] = rv
        t[t < lv] = lv
        return t

def diff_entropy_scores(X, eps=1e-6, bins=20):
    if "DataFrame" in str(type(X)):
        X = X.values
    scores = np.zeros((X.shape[1],))
    for jdx in range(0, scores.shape[0]):
        xa = X[:, jdx]
        scores[jdx] = _diff_entropy(xa, eps=eps, bins=bins)
    return scores

def _diff_entropy(x, eps=1e-6, bins=20):
    rhos, xs = np.histogram(x, density=True, bins=bins)
    xdiff = xs[1:] - xs[:-1]
    H = np.sum(rhos*np.log(rhos+eps)*xdiff)
    Hr = H/np.sum(xdiff)
    return Hr

def variance_scores(X):
    if "DataFrame" in str(type(X)):
        X = X.values
    scores = np.zeros((X.shape[1],))
    for jdx in range(0, scores.shape[0]):
        xa = X[:, jdx]
        scores[jdx] = np.var(xa)
    return scores

def q_acc_scores(X,y, q=0.5):
    qp = np.max([0.5-q, q])
    qm = np.min([1-q, q])
    cr = np.mean(y)
    if "DataFrame" in str(type(X)):
        X = X.values
    scores = np.zeros((X.shape[1],))
    for jdx in range(0, scores.shape[0]):
        xa = X[:, jdx]
        splitvalplus = np.nanquantile(xa, qp)
        splitvalmin = np.nanquantile(xa, qm)
        yl = y[np.where(X[:, jdx] > splitvalplus)]
        ys = y[np.where(X[:, jdx] <= splitvalmin)]
        scores[jdx] = np.max([np.nanmean(yl)-cr, np.nanmean(ys)-cr])/cr
    return scores



def chi2_scores(X,y, bins=10):
    if "DataFrame" in str(type(X)):
        X = X.values
    scores = np.zeros((X.shape[1],))
    for jdx in range(0, scores.shape[0]):
        xa = X[:, jdx]
        x1 = xa[y == 0]
        x2 = xa[y == 1]
        qranges = np.quantile(xa, np.arange(0, 1, 1/bins))
        scores[jdx] = chi2_score(x1, x2, qranges, bins)
    return scores

@jit
def chi2_score(x1, x2, qranges, bins):
    freq1, freq2 = np.zeros((bins,), dtype=int), np.zeros((bins,), dtype=int)
    for i in range(0, bins-1):
        freq1[i] = np.where((x1 >= qranges[i]) & (x1 < qranges[i+1]))[0].shape[0]
        freq2[i] = np.where((x2 >= qranges[i]) & (x2 < qranges[i+1]))[0].shape[0]
    freq1[i+1] = np.where(x1 >= qranges[i+1])[0].shape[0]
    freq2[i+1] = np.where(x2 >= qranges[i+1])[0].shape[0]
    return chisquare(freq1, freq2)[0]

@jit
def _cdf(x, bin_size=5):
    x = np.sort(x)
    c = len(x)
    res, _res = np.empty((0, 2)), np.empty((0, 2))
    for i in range(bin_size, c):
        if i % bin_size == 0:
            _res = np.array([i / c, np.median(x[(i - bin_size):i])])
            res = np.append(res, [_res], axis=0)
    return res


from numba.types import UniTuple, Tuple
from numba import float32 as nbfloat32, float64 as nbfloat64, int32 as nbint32
from numba import typeof
@jit(nopython=True) # , cache=True  # Tuple(nbfloat32[:], nbfloat64[:])(nbfloat32[:])
def ecdf(x):
    n = len(x)
    x = np.sort(x)
    y = np.arange(1, n+1)/n
    return x, y

# np.asarray([[i/c, np.median(x[(i-bin_size):i])] for i in range(bin_size, c) if i%bin_size==0])

@jit
def _cdfcoeff(x, bin_size=5):
    # better to take out the creation of the cdf and split the function (_cdf, _cdfcoeff)
    lt = _cdf(x, bin_size=bin_size)
    xmm = x.max() - x.min()
    diff1 = np.diff(lt[:, 1]) * lt[:-1, 0]
    diff1[0:1] = 0
    diff2 = np.diff(diff1)
    diff2[0:1] = 0
    diff2 = diff2 * lt[:-2, 0]

    # modus is peak in f'' followed by valley  in f''
    xmmp = xmm ** 4
    tv, td = (diff2[0:-1] - diff2[1:]), (np.sign(diff2[0:-1]) - np.sign(diff2[1:]))
    xrelcount, xbumpmean, xdiffmean, xdiffvar = (td > 0).sum() / (lt.shape[0] - 1), tv[
        tv > 0].mean() / xmmp, diff1.mean() / xmmp, diff1.std() / xmmp

    return xrelcount, xbumpmean, xdiffmean, xdiffvar


def _interp(xinter, yinter, xextra):
    return minterp(xinter, yinter, axis=0, extrapolate=True)(xextra)


"""
custom CDF scoring functions
"""
def cdf_scoresB(X, y, dist_type="mink_rao"):
    if "DataFrame" in str(type(X)):
        X = X.values
    scores = np.zeros((X.shape[1],))
    if len(y.shape)>1:
        for jdx in range(0, scores.shape[0]):
            scores[jdx] = _cdf_distanceB(X[np.argwhere(y==0)[:,0], jdx],
                                         X[np.argwhere(y==1)[:,0], jdx],
                                         bin_size=5,
                                         minkowski=1,
                                         dist_type=dist_type)
    else:
        for jdx in range(0, scores.shape[0]):
            scores[jdx] = _cdf_distanceB(X[y==0, jdx],
                                         X[y==1, jdx],
                                         bin_size=5,
                                         minkowski=1,
                                         dist_type=dist_type)        
    return scores

def cdf_scoresG(X, y, dist_type="emd"):
    if "DataFrame" in str(type(X)):
        X = X.values
    scores = np.zeros((X.shape[1],))
    if len(y.shape)>1:
        for jdx in range(0, scores.shape[0]):
            scores[jdx] = _cdf_distanceG(X[np.argwhere(y==0)[:,0], jdx],
                                         X[np.argwhere(y==1)[:,0], jdx],
                                         bin_size=5,
                                         dist_type=dist_type)
    else:
         for jdx in range(0, scores.shape[0]):
            scores[jdx] = _cdf_distanceG(X[y==0, jdx],
                                         X[y==1, jdx],
                                         bin_size=5,
                                         dist_type=dist_type)       
    return scores

def _cdf_distanceB(x1, x2, bin_size=5, minkowski=1, dist_type='mink_rao'):
    '''
     takes the Minkowski distance between the ecdf's and the Russell-Rao distance of the bump indicators
    '''
    lt1 = _cdf(x1, bin_size=bin_size)
    lt2 = _cdf(x2, bin_size=bin_size)

    l1 = _interp(xinter=lt1[:, 0], yinter=lt1[:, 1], xextra=lt1[:, 0])
    l2 = _interp(xinter=lt2[:, 0], yinter=lt2[:, 1], xextra=lt1[:, 0])

    l1diff = np.diff(l1)
    l2diff = np.diff(l2)

    l1diff2 = np.diff(l1diff)
    l2diff2 = np.diff(l2diff)

    l1diff2sign = np.sign(l1diff2)
    l2diff2sign = np.sign(l2diff2)

    l1bump = l1diff2sign[0:-1] - l1diff2sign[1:]
    l2bump = l2diff2sign[0:-1] - l2diff2sign[1:]

    if dist_type == 'mink_rao':
        return sc.spatial.distance.minkowski(l1, l2, p=minkowski) * sc.spatial.distance.russellrao(l1bump, l2bump)
    elif dist_type == 'mink_rao2':
        return sc.spatial.distance.minkowski(l1diff, l2diff, p=minkowski) * sc.spatial.distance.russellrao(l1bump,
                                                                                                          l2bump)
    elif dist_type == 'rao':
        return sc.spatial.distance.russellrao(l1bump, l2bump)


def _cdf_distanceG(x1, x2, bin_size=25, dist_type='emd'):
    # also see PhD-thesis Gabriel Martos Venturini, Statistical distance and probability metrics for multivariate data..etc., June 2015 Uni. Carlos III de Madrid
    # https://core.ac.uk/download/pdf/30276753.pdf
    '''
     Basically the l1-difference between the cdf's
    '''
    lt1 = _cdf(x1, bin_size=bin_size)
    lt2 = _cdf(x2, bin_size=bin_size)

    l1 = _interp(xinter=lt1[:, 0], yinter=lt1[:, 1], xextra=lt1[:, 0])
    l2 = _interp(xinter=lt2[:, 0], yinter=lt2[:, 1], xextra=lt1[:, 0])

    c = np.max([l1.max(), l2.max()]) - np.min([l1.min(), l2.min()])

    if dist_type == 'emd':
        return np.nansum(lt1[:, 0] * (np.abs(l2 - l1))) / c  # qEMD
    elif dist_type == 'cvm':
        return np.sqrt(np.nansum(lt1[:, 0] * np.power((l2 - l1), 2))) / c  # qCvM
    elif dist_type == 'cust':
        return np.nansum(lt1[:, 0] * (l2 - l1)) / c

"""
Ansatz for sequence entropy
"""

def seq_entropy_scores(X, y):
    if "DataFrame" in str(type(X)):
        X = X.values
    C = np.unique(y).shape[0]

    scores = np.zeros((X.shape[1],))
    for jdx in range(0, X.shape[1]):
        y_sorted = y[np.argsort(X[:, jdx])]
        scores[jdx] = seq_entropy(y_sorted, C)
    return scores

def seq_entropy(x, C):
    xr = x[1:]
    xl = x[:-1]
    delta = (xr == xl).astype(float)
    delta_neg = (xr != xl).astype(float)
    # F = 1./np.power(prp, len(x)+1.-2.*C)
    # return F*np.product(np.power(prp, delta)+(1-delta)*np.power(prm, delta-1))
    F = len(x) + 1. - 2. * C
    S = np.sum(delta) - np.sum(delta_neg)
    return S / F


def qseq_entropy_scores(X, y, q_type='sum', bins=20):
    if "DataFrame" in str(type(X)):
        X = X.values
    C = np.unique(y).shape[0]

    scores = np.zeros((X.shape[1],))
    for jdx in range(0, X.shape[1]):
        y_sorted = y[np.argsort(X[:, jdx])]
        scores[jdx] = qseq_entropy(y_sorted, q_type=q_type, bins=bins)
    return scores


def qseq_entropy(x, bins=20, q_type='sum'):
    di = int(len(x)/bins)
    split_arr = iter(np.split(x, np.arange(0, len(x), di)))
    next(split_arr)
    cr = np.mean(x)
    dcm = np.max([1-cr, cr])
    if q_type =='prod':
        prod = 1
        for subseq in split_arr:
            prod *= (np.abs(np.mean(subseq)-cr))/dcm
    elif q_type == 'sum':
        prod = 0
        for subseq in split_arr:
            prod += (np.abs(np.mean(subseq)-cr))/bins/dcm
    return prod

###################################################################################################

def seq_prob(n, k, p):
    div = sc.special.factorial(n) / sc.special.factorial(k) / sc.special.factorial(n - k)
    probs = np.power(p, k) * np.power(1 - p, n - k)
    return div * probs * k

'''
@jit
def get_seq_entropyX(x, seqnums, factors):
    t1 = (x == 1).astype(int)
    t0 = (x == 0).astype(int)

    res0, res1 = 0, 0
    for rownum in range(0, len(seqnums)):
        seqlen = seqnums[rownum]

        s20 = sum(np.convolve(t0, np.ones(seqlen), mode='valid') == seqlen)
        s21 = sum(np.convolve(t1, np.ones(seqlen), mode='valid') == seqlen)

        if (s20 > 0):
            f20 = factors[rownum, 0]
            res0 = res0 + s20 * f20 / len(x)
        if (s21 > 0):
            f21 = factors[rownum, 1]
            res1 = res1 + s21 * f21 / len(x)
        if (s20 + s21) == 0:
            return np.nanmax([res0, res1])
    return np.nanmax([res0, res1])

@jit
def seq_entropyX_scores(X, y, seqrange=(2, 20)):
    if "DataFrame" in str(type(X)):
        X = X.values
    scores = np.zeros((X.shape[1],))
    p = np.mean(y)

    factors = np.zeros((seqrange[1] - seqrange[0], 2))
    seqnums = np.zeros((seqrange[1] - seqrange[0],), dtype='int64')
    for idx, seqlen in enumerate(range(seqrange[0], seqrange[1])):
        seqnums[idx] = seqlen
        factors[idx, 0] = 1 / seq_prob(seqlen, seqlen, (1 - p))
        factors[idx, 1] = 1 / seq_prob(seqlen, seqlen, (p))

    for jdx in range(0, scores.shape[0]):
        xa = X[:, jdx]
        scores[jdx] = get_seq_entropyX(y[np.argsort(xa)], seqnums, factors)
    return scores
'''
# alternative with find_runs

#@jit
def _seq_prob(k, p):
    n = k
    div = sc.special.factorial(n) / sc.special.factorial(k) / sc.special.factorial(n - k)
    probs = np.power(p, k) * np.power(1 - p, n - k)
    return div * probs * k

#@jit
def seq_entropyX_scores(X, y):
    if "DataFrame" in str(type(X)):
        X = X.values
    scores = np.zeros((X.shape[1],))
    numsamples = len(y)
    cprob = np.max(np.bincount(y)/numsamples)

    for jdx in range(0, scores.shape[0]):
        xa = X[:, jdx]
        res = find_runs(y[np.argsort(xa)])[2]
        counts = np.bincount(res)
        seqlen = np.arange(len(counts))
        probs = cprob * np.ones(len(counts)-1,)
        factors = 1/np.apply_along_axis(_seq_prob, 0, seqlen[1:], probs)
        scores[jdx] = np.log10(sum(factors*counts[1:])/numsamples)
    return scores


"""
Variance distances
"""
def var_dists(X, y):
    if "DataFrame" in str(type(X)):
        X = X.values
    scores = np.zeros((X.shape[1],))
    if len(y.shape)>1:
        for jdx in range(0, scores.shape[0]):
            #gvar = np.var(X[:, jdx])
            scores[jdx] = var_dist(X[np.argwhere(y==0)[:, 0], jdx],
                               X[np.argwhere(y==1)[:, 0], jdx])
    else:
         for jdx in range(0, scores.shape[0]):
            #gvar = np.var(X[:, jdx])
            scores[jdx] = var_dist(X[y==0, jdx],
                                   X[y==1, jdx])       
    return scores

@jit
def var_dist(x1, x2):
    return (np.var(x2)-np.var(x1))

"""
Quantile distances
"""

def q_dists(X, y, q=0.5):
    if "DataFrame" in str(type(X)):
        X = X.values
    scores = np.zeros((X.shape[1],))
    if len(y.shape)>1:
        for jdx in range(0, scores.shape[0]):
            scores[jdx] = q_dist(X[np.argwhere(y==0)[:, 0], jdx],
                                 X[np.argwhere(y==1)[:, 0], jdx],
                                 q=q)
    else:
        for jdx in range(0, scores.shape[0]):
            scores[jdx] = q_dist(X[y==0, jdx],
                                 X[y==1, jdx],
                                 q=q)        
    return scores

def q_dist(x1, x2, q=0.5, weighted=True):
    q1 = np.nanquantile(x1, q)
    q2 = np.nanquantile(x2, q)
    if weighted:
        q1std = np.nanstd(x1)
        q2std = np.nanstd(x2)
        return (q2 - q1)/np.min([q1std, q2std])
    else:
        return q2 - q1


"""
Probability of exceedence
"""

def _exceed_score(xl, xr, xa):
    ecdf1 = ecdf(xl)
    ecdf2 = ecdf(xr)
    ecdfT = ecdf(xa)
    _prob = (np.dot(ecdf2[0], ecdf2[1]) - np.dot(ecdf1[0], ecdf1[1])) / \
            np.dot(ecdfT[0], ecdfT[1])*2+0.5
    prob_one_is_larger = np.min([1, np.max([_prob, 1 - _prob])])
    return 2*(prob_one_is_larger-0.5)

def _raw_exceed_score(xl, xr, n_samples=500):
    xls = np.random.choice(xl, n_samples)
    xrs = np.random.choice(xr, n_samples)
    _prob = sum(xls > xrs)/n_samples
    prob_one_is_larger = np.min([1, np.max([_prob, 1 - _prob])])
    return 2*(prob_one_is_larger-0.5)

def prob_exceed_scores(X, y):
    if "DataFrame" in str(type(X)):
        X = X.values
    scores = np.zeros((X.shape[1],))
    np.random.seed(1234)
    if len(y.shape)>1:
        for jdx in range(0, scores.shape[0]):
            xa = X[:, jdx]
            xl = X[np.argwhere(y == 0)[:, 0], jdx]
            xr = X[np.argwhere(y == 1)[:, 0], jdx]
            scores[jdx] = _exceed_score(xl, xr, xa)
    else:
        for jdx in range(0, scores.shape[0]):
            xa = X[:, jdx]
            xl = X[y == 0, jdx]
            xr = X[y == 1, jdx]
            scores[jdx] = _exceed_score(xl, xr, xa)        
    return scores


"""
Monotonic alignment
"""
def monotonic_alignment(X, y, return_df=False):
    if "DataFrame" in str(type(X)):
        inds = X.columns
        X = X.values
    else:
        inds = np.arange(0, X.shape[1])
    cols = ['mono_align']

    ydiff = np.diff(y, axis=0)
    Xdiff = np.diff(X, axis=0)

    scores = np.zeros((X.shape[1]))
    for jdx in range(0, X.shape[1]):
        _x = np.sign(Xdiff[:, jdx])
        _y = np.sign(ydiff[:])
        scores[jdx] = np.dot(_x, _y)/(len(y)-1)
    if return_df:
        return pd.DataFrame(data=scores, index=inds, columns=cols)
    else:
        return scores


"""
Pearson applied to array
"""
def pearson_scores(X, y, return_df=False, correction=None, loglog=False):
    if "DataFrame" in str(type(X)):
        inds = X.columns
        X = X.values
    else:
        inds = np.arange(0, X.shape[1])
    if loglog:
        cols = ['pearson_loglog_score', 'pearson_loglog_pval']
    else:
        cols = ['pearson_score', 'pearson_pval']

    #C = np.unique(y).shape[0]
    scores = np.zeros((X.shape[1], 2))
    eps = 1
    yv = y + np.abs(np.min(y)) * (1 - np.sign(np.min(y)))*0.5 + eps

    for jdx in range(0, X.shape[1]):
        if loglog:
            xv = X[:, jdx]
            xv = xv + np.abs(np.min(xv))*(1 - np.sign(np.min(xv)))*0.5 + eps
            _x = np.log(xv)
            _y = np.log(yv)
        else:
            _x = X[:, jdx]
            _y = y[:]
        scores[jdx, :] = pearsonr(_x, _y)
    if correction=='bonferroni':
        dim = X.shape[1]
        scores[:, 1] =  scores[:, 1]*dim # 1-np.power(1-scores[:, 1], dim)
    if return_df:
        return pd.DataFrame(data=scores, index=inds, columns=cols)
    else:
        return scores

"""
Spearman applied to array
"""

def spearman_scores(X, y, return_df=False, correction=None):
    if "DataFrame" in str(type(X)):
        inds = X.columns
        X = X.values
        try:
            y = y.values
        except:
            pass
    else:
        inds = np.arange(0, X.shape[1])
    cols = ['spearman_score', 'spearman_pval']

    #C = np.unique(y).shape[0]
    scores = np.zeros((X.shape[1], 2))
    for jdx in range(0, X.shape[1]):
        sorted_ind = np.argsort(X[:, jdx])
        x_sorted = X[sorted_ind, jdx]
        y_sorted = y[sorted_ind]
        scores[jdx, :] = spearmanr(x_sorted, y_sorted)
    if correction=='bonferroni':
        dim = X.shape[1]
        scores[:, 1] =  scores[:, 1]*dim #1-np.power(1-scores[:, 1], dim)
    if return_df:
        return pd.DataFrame(data=scores, index=inds, columns=cols)
    else:
        return scores


"""
Kolmogorov-Smirnov applied to array
"""

def ks_scores(X,y):
    if "DataFrame" in str(type(X)):
        X = X.values
    scores = np.zeros((X.shape[1], 2))
    if len(y.shape)>1:
        for jdx in range(0, scores.shape[0]):
            scores[jdx,:] = ks(X[np.argwhere(y==0)[:,0], jdx], 
                               X[np.argwhere(y==1)[:,0], jdx])
    else:
        for jdx in range(0, scores.shape[0]):
            scores[jdx,:] = ks(X[y==0, jdx], 
                               X[y==1, jdx])        

    return scores



"""
Wasserstein 1 distance applied to array
"""
def wass1_scores(X,y):
    if "DataFrame" in str(type(X)):
        X = X.values
    scores = np.zeros((X.shape[1],))
    if len(y.shape)>1:
        for jdx in range(0, scores.shape[0]):
            scores[jdx] = wass(X[np.argwhere(y==0)[:,0], jdx], 
                               X[np.argwhere(y==1)[:,0], jdx])
    else:
        for jdx in range(0, scores.shape[0]):
            scores[jdx] = wass(X[y==0, jdx], 
                               X[y==1, jdx])        
    return scores


"""
entropy divergence applied to array
"""
def ec_scores(X,y, num_bins=25, ent_type='kl'):
    if "DataFrame" in str(type(X)):
        X = X.values
    scores = np.zeros((X.shape[1],))

    if len(y.shape)>1:
        for jdx in range(0, scores.shape[0]):
            scores[jdx] = _information_change(X[np.argwhere(y==0)[:,0], jdx], 
                               		          X[np.argwhere(y==1)[:,0], jdx], num_bins=num_bins, ent_type='kl')
    else:
        for jdx in range(0, scores.shape[0]):
            scores[jdx] = _information_change(X[y==0, jdx], 
                                              X[y==1, jdx], num_bins=num_bins, ent_type='kl')
    return scores   


"""
     Information change: 
        * Kullback-Leibler divergence (have to make same number of bins)
        * cross-entropy
        * Shannon entropy change
"""
def _information_change(v1, v2, ent_type = 'kl', bin_type='fixed', num_bins=10):
    '''
    v1: vector one
    v2: vector two
    ent_type : kl, shannon, cross, js (jensen-shannon)
    bin_type : auto, fixed
    num_bins : non-zero positive integer
    
    return entropy difference
    '''
    # get bins
    num_bins = num_bins if bin_type=='fixed' else bin_type
    v1bins = np.histogram(v1, density=True, bins=num_bins)
    v2bins = np.histogram(v2, density=True, bins=num_bins)
    
    ent1 = -np.sum(v1bins[0]*np.log2(v1bins[0]))
    ent2 = -np.sum(v2bins[0]*np.log2(v2bins[0]))    
    
    log2v1 = np.log2(v1bins[0])
    log2v2 = np.log2(v2bins[0])
    
    if ent_type == 'shannon':
        return 2*np.abs(ent1-ent2)/(np.abs(ent1)+np.abs(ent2))
    elif ent_type == 'cross':
        cross1 = -np.sum(v1bins[0]*log2v2)
        cross2 = -np.sum(v2bins[0]*log2v1)        
        return 2*np.max([np.abs(cross1),np.abs(cross2)])/(np.abs(ent1)+np.abs(ent2))
    elif ent_type == 'kl':
        return 2*np.max([np.abs(np.sum(v1bins[0]*(log2v1-log2v2))),
                         np.abs(np.sum(v2bins[0]*(log2v2-log2v1)))])/(np.abs(ent1)+np.abs(ent2))
    elif ent_type == 'js': 
        vbins = np.histogram(np.hstack([v1,v2]), density=True, bins=num_bins)
        log2v = np.log2(vbins[0])
        return 0.5*(np.abs(np.sum(v1bins[0]*(log2v1-log2v))) + np.abs(np.sum(v2bins[0]*(log2v2-log2v))))/(np.abs(ent1)+np.abs(ent2))


'''
UPDATE: Cross-entropy/KL-divergence/Jensen-Shannon
'''

"""
entropy divergence applied to array
"""
def ec_scores2(X,y, num_bins=25, ent_type='kl'):
    '''
    :param X: ..
    :param y: ..
    :param num_bins:
    :param ent_type: kl, js, cross
    :return:
    '''
    eps = 1e-6

    if "DataFrame" in str(type(X)):
        X = X.values
    scores = np.zeros((X.shape[1],))
    for jdx in range(0, scores.shape[0]):
        xa = X[:, jdx]
        xl = xa[np.argwhere(y == 0)]
        xr = xa[np.argwhere(y == 1)]

        hist_ranges = np.histogram(xa, num_bins, density=True)[1]
        xlp = np.histogram(xl, bins=hist_ranges, density=True)[0] + eps
        xrp = np.histogram(xr, bins=hist_ranges, density=True)[0] + eps
        if ent_type=='kl':
            scores[jdx] = _kl_divergence(xlp, xrp)
        elif ent_type=='js':
            scores[jdx] = _js_divergence(xlp, xrp)
        elif ent_type=='cross':
            scores[jdx] = _cross_entropy(xlp, xrp)
    return scores

def _kl_divergence(x1p, x2p):
    return np.nansum(sc.special.kl_div(x1p, x2p))

def _js_divergence(x1p, x2p):
    return sc.spatial.distance.jensenshannon(x1p, x2p)

def _cross_entropy(x1p,x2p):
    return sc.stats.entropy(x1p, x2p)

################################################################################

def ecs_scores(X,y, num_bins=21, ent_type='kl', num_sample_rounds=31):
    '''
    :param X: ..
    :param y: ..
    :param num_bins:
    :param ent_type: kl, js, cross
    :return:
    '''
    eps = 1e-6
    num_class = 2 # np.bincount(y).shape[0]
    ###### HACK #######################
    ##### REPLACE Y==2 by random select of classes 1 and 0
    y = np.where(y == 2, np.random.randint(0, 2, len(y)), y)
    ###################################
    ##### CODE HELL AWAITS.. ##########

    if "DataFrame" in str(type(X)):
        X = X.values
    scores = np.zeros((X.shape[1],))
    for jdx in range(0, scores.shape[0]):
        xa = X[:, jdx]
        ys = y[np.argsort(xa)]

        probs = np.zeros((num_sample_rounds * num_class, 3))
        adx = 0
        bdx = num_class
        for m in range(0, num_sample_rounds):
            np.random.shuffle(y)
            yr = np.random.choice(y, num_bins)
            yref = np.random.choice(y, num_bins)

            ysortrand = np.random.choice(ys, num_bins)

            yrprobs = np.bincount(yr) / num_bins + eps
            yrfprobs = np.bincount(ysortrand) / num_bins + eps
            yrefprobs = np.bincount(yref) / num_bins + eps

            probs[adx:bdx, 0] = yrprobs
            probs[adx:bdx, 1] = yrfprobs
            probs[adx:bdx, 2] = yrefprobs
            adx += num_class
            bdx += num_class

        if ent_type=='kl':
            scores[jdx] = _kl_divergence(probs[:, 0], probs[:, 1])/_kl_divergence(probs[:, 0], probs[:, 2])
        elif ent_type=='js':
            scores[jdx] = _js_divergence(probs[:, 0], probs[:, 1])/_js_divergence(probs[:, 0], probs[:, 2])
        elif ent_type=='cross':
            scores[jdx] = _cross_entropy(probs[:, 0], probs[:, 1])/_cross_entropy(probs[:, 0], probs[:, 2])
    return scores

###############################################################################################################
###############################################################################################################

class fs_ws1():
    scores_ = None

    def __init__(self, pvalue=0.01):
        self.pvalue = pvalue

    def apply_test(self, pos, neg, column):
        zscore = wass(pos[:, column], neg[:, column])
        return zscore

    def fit(self, x, y):
        zero_idx = np.where(y == 0)[0]
        one_idx = np.where(y == 1)[0]
        pos_samples = x[one_idx]
        neg_samples = x[zero_idx]
        results_ = np.array(list(map(lambda c:
                                     self.apply_test(pos_samples, neg_samples, c), range(0, x.shape[1]))))
        self.scores_ = results_
        return self


class fs_ws2():
    scores_ = None

    def __init__(self, pvalue=0.01):
        self.pvalue = pvalue

    def apply_test(self, pos, neg, column):
        zscore = energy_distance(pos[:, column], neg[:, column])
        return zscore

    def fit(self, x, y):
        zero_idx = np.where(y == 0)[0]
        one_idx = np.where(y == 1)[0]
        pos_samples = x[y == 0]
        neg_samples = x[y == 1]
        results_ = np.array(list(map(lambda c:
                                     self.apply_test(pos_samples, neg_samples, c), range(0, x.shape[1]))))
        self.scores_ = results_
        return self


class fs_mannwhitney():
    pvalues_ = None
    scores_ = None
    results_ = None

    def __init__(self, pvalue=0.01, mode='auto'):
        # mode : 'auto', 'exact', 'asymp'
        self.pvalue = pvalue
        self.mode = mode

    def apply_test(self, pos, neg, column):
        zscore, p_value = mwu(pos[:, column], neg[:, column], alternative="less")  # mode=self.mode
        return zscore, p_value

    def fit(self, x, y):
        zero_idx = np.where(y == 0)[0]
        one_idx = np.where(y == 1)[0]
        pos_samples = x[one_idx]
        neg_samples = x[zero_idx]
        self.results_ = np.array(list(map(lambda c:
                                          self.apply_test(pos_samples, neg_samples, c), range(0, x.shape[1]))))
        self.scores_ = self.results_[:, 0]
        self.pvalues_ = self.results_[:, 1]
        return self

    def transform(self, x):
        not_signif = self.p_values < self.pvalue
        to_delete = [idx for idx, item in enumerate(not_signif) if item == False]
        return np.delete(x, to_delete, axis=1), to_delete


class fs_ks():
    pvalues_ = None
    scores_ = None
    results_ = None

    def __init__(self, pvalue=0.01):
        self.pvalue = pvalue

    def apply_test(self, pos, neg, column):
        zscore, p_value = ks2(pos[:, column], neg[:, column])
        return zscore, p_value

    def fit(self, x, y):
        zero_idx = np.where(y == 0)[0]
        one_idx = np.where(y == 1)[0]
        pos_samples = x[one_idx]
        neg_samples = x[zero_idx]
        self.results_ = np.array(list(map(lambda c:
                                          self.apply_test(pos_samples, neg_samples, c), range(0, x.shape[1]))))
        self.scores_ = self.results_[:, 0]
        self.pvalues_ = self.results_[:, 1]
        return self

    def transform(self, x):
        not_signif = self.p_values < self.pvalue
        to_delete = [idx for idx, item in enumerate(not_signif) if item == False]
        return np.delete(x, to_delete, axis=1), to_delete


class fs_epps():
    pvalues_ = None
    scores_ = None
    results_ = None

    def __init__(self, pvalue=0.01):
        self.pvalue = pvalue

    def apply_test(self, pos, neg, column):
        try:
            zscore, p_value = epps(pos[:, column], neg[:, column])
            return zscore, p_value
        except Exception as e:
            return np.nan, np.nan 

    def fit(self, x, y):
        zero_idx = np.where(y == 0)[0]
        one_idx = np.where(y == 1)[0]
        pos_samples = x[one_idx]
        neg_samples = x[zero_idx]
        self.results_ = np.array(list(map(lambda c:
                                          self.apply_test(pos_samples, neg_samples, c), 
                                          range(0, x.shape[1]))))
        self.scores_ = self.results_[:, 0]
        self.pvalues_ = self.results_[:, 1]
        return self

    def transform(self, x):
        not_signif = self.p_values < self.pvalue
        to_delete = [idx for idx, item in enumerate(not_signif) if item == False]
        return np.delete(x, to_delete, axis=1), to_delete

#######################################################################################################################
#######################################################################################################################



#######################################################################################################################
#######################################################################################################################


def _cov(x, lib='numpy', method='empirical', inverse=False):
    # lib: numpy, scipy, sklearn
    # normalised: True/false  np.corrcoef if true
    # method: exact/shrunk/sparse/empirical
    
    assert method in ['shrunk', 'sparse', 'empirical']
    assert lib in ['numpy', 'scipy', 'sklearn']
    
    if method=='empirical':    
        if (lib=='numpy') | (normalised):
            if normalised:
                cov = np.corrcoef(x)
            else:
                cov = np.cov(x)
            if inverse:
                invcov = np.linalg.inv(cov)
        elif lib=='scipy':
            cov = sc.stats.cov(x)
            if inverse:
                invcov = np.linalg.inv(cov)
        elif lib=='sklearn':
            cm = EmpiricalCovariance().fit(x)
            cov = cm.covariance_
            if inverse:
                invcov = cm.precision_
    else:
        if method=='shrunk':
            # return LedoitWolf().fit(x).covariance_
            cm = EmpiricalCovariance().fit(x)
            cov = cm.covariance_
            invcov = cm.precision_
        elif method=='sparse':
            cm = GraphicalLassoCV().fit(x) 
            cov = cm.covariance_
            invcov = cm.precision_
    
    if inverse:
        return cov, invcov
    else:
        return cov, None

@jit
def _Mahalanobis(v1, v2, inv_cov):
    return sc.spatial.distance.mahalanobis(v1, v2, inv_cov)

def Mahalanobis(X):
    cov, icov = _cov(X)

@jit
def _skewness(x, logscale=False, bound=False, scale=1000, sample=False, bias=True):
    '''
    x : array(N,1)
    return skewness
    '''
    std = np.std(x)
    mu = np.mean(x)
    N = x.shape[0]
    val = np.sum(np.power(x-mu, 3))/np.power(std, 3)
    if sample==False:
        if bound==False:
            if logscale==False:
                return val
            else:
                return np.log10(np.abs(val))*np.sign(val)
        else:
            return np.tanh(val/scale)
    else:
        return stats.skew(x, bias=bias)

@jit
def _kurtosis(x, logscale=False, sample=False, bias=True):            
    std = np.std(x)
    mu = np.mean(x) 
    val = np.sum(np.power(x-mu, 4))/np.power(std, 4)
    if sample==False:
        if logscale==False:
            return val
        else:
            return np.log10(np.abs(val))*np.sign(val)
    else:
        return stats.kurtosis(x, bias=bias, nan_policy='omit')

@jit
def _stanmom(x, mom=3, logscale=False, mutype=0):
    # mom : 3 is skewness, 4 is kurtosis
    std = np.std(x)
    if mutype==0:
        mu = np.mean(x)
    else:
        mu = np.median(x)
    val = np.sum(np.power(x-mu, mom))/np.power(std, mom)
    if logscale==False:
        return val
    else:
        return np.log10(np.abs(val))*np.sign(val)

@jit
def _rjb(x, C1=6, C2=64):
    n = len(x)
    mu3 = _stanmom(x, mom=3, logscale=False)
    mu4 = _stanmom(x, mom=4, logscale=False)
    J = np.sqrt(np.pi/2.)*np.sum(np.abs(x-np.median(x, axis=0)))
    res = n/C1*(mu3/J**3)**2 + n/C2*(mu4/J**4-3)**2
    return res

@jit
def fisher_criterion(v1, v2):
    N = m1.shape[0]
    m1, m2 = np.mean(v1), np.mean(v2)
    s1, s2 = np.var(v1), np.var(v2)
    return np.abs(m1-m2)/(s1+s2)


    
from unidip import UniDip  
import unidip.dip as dip
def _multimodality(x, method='hartigan'):
    x = np.msort(x)
    # multi-modality can be extracted from the number of inflection points on a q-q plot 
    if method=='hartigan':
        return len(UniDip(x).run())
    
def _qq(x, cdist=stats.norm, minkowski=2, plot=None):
    # surface area,max_slope,min_slope
    
    if (len(x.shape)==1):
        vref, v = stats.probplot(x, dist=cdist, plot=plot, fit=False)
        multi=False
    else:
        if x.shape[1]==1:
            vref, v = stats.probplot(x, dist=cdist, plot=plot, fit=False)
        else:
            _cols = x.columns.tolist()
            x = np.array(x)
            vref, v= np.zeros(shape=(x.shape[1], x.shape[0])), np.zeros(shape=(x.shape[1], x.shape[0]))
            for j in range(0, x.shape[1]):
                vref[j, :], v[j, :] = stats.probplot(x[:,j], dist=cdist, plot=None, fit=False)
            multi=True
            
    if multi==False:
        dist_euc = sc.spatial.distance.minkowski(vref, v, p=minkowski)
        mm0 = np.array(np.max(vref)-np.min(vref))
        mm1 = np.array(np.max(v)-np.min(v))
        nsam  = mm0*mm1*0.5
        asam = 0.5*np.sum(np.diff(vref)*(v[:-1]+v[1:]))
        sm = (asam-nsam)/nsam 
        return 0.5*dist_euc/mm0/mm1*sm**2
    else:
        dist_euc = np.diag(sc.spatial.distance.cdist(vref , v, metric='minkowski', p=minkowski))  
        mm0 = np.array(np.max(vref, axis=1)-np.min(vref, axis=1))
        mm1 = np.array(np.max(v, axis=1)-np.min(v, axis=1))
        nsam  = mm0*mm1*0.5
        asam = 0.5*np.sum(np.diff(vref, axis=1)*(v[:, :-1]+v[:, 1:]), axis=1)
        sm = (asam-nsam)/nsam 
        res = np.reshape(0.5*dist_euc/mm0/mm1*sm**2, (1,-1))
        return pd.DataFrame(res, columns=_cols)
    
def _qq2(x1, x2, qbins=15, plot=False, minkowski=2):
    # identify left-skewness, right-skewness, under-dispersed, over-dispersed data
    # see e.g. http://www.ucd.ie/ecomodel/Resources/QQplots_WebVersion.html
    # surface area,max_slope,min_slope
    
    qbins = np.min([qbins, np.min([len(x1), len(x2)])])
    qstep = np.max([1, int(100/qbins)])
    
    perc1 = np.percentile(x1, q=np.arange(0, 100, qstep))
    perc2 = np.percentile(x2, q=np.arange(0, 100, qstep))
    
    if plot:
        plt.plot(perc1, perc2)
        plt.plot([np.min(perc1), np.max(perc1)],[np.min(perc2), np.max(perc2)])
        plt.title('Q-Q')  
        plt.xlabel('q-'+meta_vals[0])
        plt.ylabel('q-'+meta_vals[1])
    
    #dist_cos = sc.spatial.distance.cosine(perc1, perc2)
    dist_euc = sc.spatial.distance.minkowski(perc1, perc2, p=minkowski)
    mm0 = np.max(perc1)-np.min(perc1)
    mm1 = np.max(perc2)-np.min(perc2)
    nsam  = mm0*mm1*0.5
    asam = 0.5*np.sum(np.diff(perc1)*(perc2[:-1]+perc2[1:]))
    # skewed metric
    # positive: left-skewed, negative: right-skewed
    sm = (asam-nsam)/nsam 
    return dist_euc/mm0/mm1*sm**2

@jit
def wcorr(x1, x2, w):
    '''
    x1, x2 : array of size (N samples,) or (N, m)
    weights : weight of each sample/feature
    returns the weighted Pearson correlation
    '''
    if w is None:
        w = np.ones(len(x1))

    mu1=np.mean(x1)
    mu2=np.mean(x2)
    xdv = x1-mu1
    yvd = x2-mu2
    c = np.sum(w*xdv*yvd)/np.sqrt(np.sum(w*xdv**2)*np.sum(w*yvd**2))
    return c

def _corr(x1, x2, fun=None):
    '''
    x1, x2 : array of size (N samples,) or (N, m)
    fun: is function object that takes x1,x2,weights as input, e.g. spearmanr, pearsonr, f_oneway
    '''    
    return fun(x1,x2)

def _continuous_sim(x1, x2, fun=None, centered=False, w=None, minkowski=2):
    '''
    x1, x2 : array of size (N samples,) or (N, m)
    fun : chebyshev, cityblock, cosine, euclidean, jensenshannon ([0,1]), mahalanobis, minkowski, canberra, braycurtis
    centered : if True centers x1, x2 around the mean
    weights : weight of each sample/feature
    '''    
    if centered:
        x1 = x1-np.mean(x1)
        x2 = x2-np.mean(x2)
    if fun==None:
        fun = sc.spatial.distance.euclidean
        return fun(x1,x2, w=w)
    else:
        if fun.__name__ == 'minkowski':
            return fun(x1,x2, w=w, p=minkowski)
        else:
            return fun(x1,x2, w=w)

def boolean_sim(x1, x2, fun=None, w=None):
    '''
    x1, x2 : array of size (N samples,) or (N, m)
    fun : is function object that takes x1,x2,weights as input, e.g. hamming, jaccard, russellrao, kulsinski, sokalmichener, sokalsneath
    weights : weight of each sample/feature
    '''
    if fun==None:
        fun = sc.spatial.distance.hamming    
    return fun(x1,x2,w)

def stat_test_arr(x, test='AD', rescale=False):
    '''
     wrapper to enable application of scipy statistical tests in groupby calls
     test : 'AD', 'KS', 'SW', 'KS'
    '''
    #x = np.array(x)        
    if test == 'AD':
        fun = sc.stats.anderson
        corr = 1
    elif test == 'SW':
        fun = sc.stats.shapiro
        corr = 10
    elif test == 'JB':
        fun = sc.stats.jarque_bera
        corr = 1e5
    elif test == 'KS':
        fun = sc.stats.kstest
        corr = 5
    
    res = np.zeros((1,x.shape[1])) 
    if test in ['AD', 'SW', 'JB']:
        for j in range(0,x.shape[1]):
            if rescale==False:
                res[0,j] = fun(x.iloc[:,j])[0]
            else:
                res[0,j] = np.tanh(fun(x.iloc[:,j])[0]*corr)
    else:
        for j in range(0,x.shape[1]):
            if rescale==False:
                res[0,j] = fun(x.iloc[:,j], 'norm')[0]
            else:
                res[0,j] = np.tanh(fun(x.iloc[:,j], 'norm')[0]*corr)
    return pd.DataFrame(data=res, columns=x.columns)

    
def _scaled_stat_distance(x1, x2, test='KS'):
    # scaled with difference between median's
    mdiff = np.median(x2)-np.median(x1)
    return mdiff*stats.ks_2samp(x1, x2)[0]


def search_sequence_numpy(arr, seq):
    """ Find sequence in an array using NumPy only.

    Parameters
    ----------
    arr    : input 1D array
    seq    : input 1D array

    Output
    ------
    Output : 1D Array of indices in the input array that satisfy the
    matching of input sequence in the input array.
    In case of no match, an empty list is returned.
    SOURCE/CREDITS : https://stackoverflow.com/questions/36522220/searching-a-sequence-in-a-numpy-array
    """

    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na - Nseq + 1)[:, None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any() > 0:
        return np.where(np.convolve(M, np.ones((Nseq), dtype=int), mode='valid') > 0)[0]
    else:
        return []

#@jit
def find_runs(x):
    """Find runs of consecutive items in an array.

    SOURCE/CREDITS : https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
    """

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths

#######################################################################################################################
#recursive feature splitter
#######################################################################################################################

def _entropy(x):
    return True

def _information_gain(x, xs):
    return True

class univariate_feature_splitter():
    '''
    Finds a list of feature-ranges with entropy < threshold and of a minimum length
    '''
    def __init__(self, max_depth=2, min_bin_size=10):
        self.max_depth = max_depth
        self.min_bin_size = min_bin_size

    def fit(self, x, y):
        # find split in x with maximum information gain

        self.splits, self.entropy
        return True

    def predict(self, x):
        return y_pred

    def _score(self):
        return True

#######################################################################################################################
# association rule miner
#######################################################################################################################

from itertools import combinations, groupby
from collections import Counter

class association_ruler():
    '''
    Credits go to Grace Tenorio for 99% of the code: https://www.datatheque.com/posts/association-analysis/

    also: https://stackoverflow.com/questions/36676576/map-a-numpy-array-of-strings-to-integers

    Based on the A-priori algorithm
    '''

    def __init__(self, num_bins=5, plot=False, debug=False):
        self.num_bins = num_bins
        self.debug= False
        self.plot = False


    def pre_assoc_array(self, X, num_bins=None):
        '''
        This function turns the array with expression values in 'baskets' that can be easily plugged in 
        association-rule mining algorithms.

        X: data
        num_bins: number of quantile bins to categorize the expression data

        returns: np.array([[sample_1, feature_1_bin_0], [sample_1, feature_2_bin_2],...])

        -> out.shape = (M*N, 2)
        ''' 

        assert "DataFrame" in str(type(X)), "For now we only allow dataframes"
        assert X.shape[1] == len(set(X.columns)), "There are duplicate column names"

        if num_bins is None:
            num_bins = self.num_bins
                                  
        old_columns = X.columns
        new_X = X.copy()
        for idx, _col in enumerate(X.columns):
            new_col = _col+str(idx)
            new_X[new_col] = pd.qcut(X[_col], q=num_bins, labels=False, duplicates='drop')
        new_X = new_X.drop(old_columns, axis=1)


        new_X['id']= new_X.index
        new_X=pd.melt(new_X, id_vars='id')
        new_X.set_index('id', inplace=True)
        new_X.dropna(subset=['value'], axis=0, inplace=True)
        new_X['value'] = new_X.value.astype(int)
        new_X['val']= new_X.apply(lambda x: str(x[0])+"_"+str(x[1]), axis=1)
        new_X.drop(new_X.columns[[0,1]], axis=1, inplace=True)


        # mappings
        new_X.sort_index(inplace=True)
        lookupTable_item, indexed_dataSet_item = np.unique(new_X.val, return_inverse=True)
        lookupTable_sample, indexed_dataSet_sample = np.unique(new_X.index, return_inverse=True)

        new_X['val'] = indexed_dataSet_item
        new_X.set_index(indexed_dataSet_sample, inplace=True)

        map_item_tuple = (lookupTable_item, indexed_dataSet_item)
        map_sample_tuple =  (lookupTable_sample, indexed_dataSet_sample)

        self.map_item_dict = dict(zip(map_item_tuple[1], map_item_tuple[0]))
        self.map_sample_dict = dict(zip(map_sample_tuple[1], map_sample_tuple[0]))

        if self.debug:
            self.new_X = new_X

        return new_X['val']

    # Returns frequency counts for items and item pairs
    def freq(self, iterable):
        if type(iterable) == pd.core.series.Series:
            return iterable.value_counts().rename("freq")
        else: 
            return pd.Series(Counter(iterable)).rename("freq")

        
    # Returns number of unique orders
    def order_count(self, order_item):
        return len(set(order_item.index))


    # Returns generator that yields item pairs, one at a time
    def get_item_pairs(self, order_item):
        order_item = order_item.reset_index().values
        if self.debug:
            self.debug_order_item4 = order_item
        for order_id, order_object in groupby(order_item, lambda x: x[0]):
            item_list = [item[1] for item in order_object]
                  
            for item_pair in combinations(item_list, 2):
                yield item_pair
                

    # Returns frequency and support associated with item
    def merge_item_stats(self, item_pairs, item_stats):
        return (item_pairs
                    .merge(item_stats.rename(columns={'freq': 'freqA', 'support': 'supportA'}), left_on='item_A', right_index=True)
                    .merge(item_stats.rename(columns={'freq': 'freqB', 'support': 'supportB'}), left_on='item_B', right_index=True))


    # Returns name associated with item
    def merge_item_name(self, rules, item_name):
        columns = ['itemA','itemB','freqAB','supportAB','freqA','supportA','freqB','supportB', 
                   'confidenceAtoB','confidenceBtoA','lift']
        rules = (rules
                    .merge(item_name.rename(columns={'item_name': 'itemA'}), left_on='item_A', right_on='item_id')
                    .merge(item_name.rename(columns={'item_name': 'itemB'}), left_on='item_B', right_on='item_id'))
        return rules[columns]               



    def association_rules(self, order_item, min_support=10, pre_ordered=False, sample_column=None, debug=False):
        '''
        If pre_ordered we assume that order_item has the form pd.Dataframe([[index, item], [index, item2]...]]),
        where e.g. in the case of expression data the index would be the sample_id, and the item would be the feature bin's for that sample.
        '''
        if pre_ordered:
            assert "Series" in str(type(order_item)), "For now we only allow Series for the pre_ordered data"
        else:
            assert "DataFrame" in str(type(order_item)), "For now we only allow Dataframes for the non-ordered data"
        if pre_ordered:
            assert order_item.shape[1]==1, "Your pre-ordered dataframe should only contain 1 column, \
                the index column represents the sample_id or the order_id, the 1 column represents the \
                feature bin or the product id"
        else:
            if sample_column is not None:
                assert sample_column in order_item.columns.tolist(), "You explicitly set the sample_column \
                    but it is not present in the dataframe"
                order_item.set_index(sample_column, inplace=True)

        if debug:
            self.debug=True

        if ~pre_ordered:
            print("Creating item/order array")
            order_item = self.pre_assoc_array(order_item)

        print("Starting order_item: {:22d}".format(len(order_item)))


        # Calculate item frequency and support
        item_stats             = self.freq(order_item).to_frame("freq")
        order_count           = self.order_count(order_item)
        item_stats['support']  = item_stats['freq'] / order_count * 100

        if debug:
            self.debug_item_stats1 = item_stats
            self.debug_order_count = order_count
            self.debug_order_item1 = order_item

        # Filter from order_item items below min support 
        qualifying_items       = item_stats[item_stats['support'] >= min_support].index
        order_item             = order_item[order_item.isin(qualifying_items)]

        if debug:
            self.debug_qualifying_items = qualifying_items
            self.debug_order_item2 = order_item

        print("Items with support >= {}: {:15d}".format(min_support, len(qualifying_items)))
        print("Remaining order_item: {:21d}".format(len(order_item)))

        # Filter from order_item orders with less than 2 items
        order_size             = self.freq(order_item.index)
        qualifying_orders      = order_size[order_size >= 2].index
        order_item             = order_item[order_item.index.isin(qualifying_orders)]

        if debug:
            self.debug_qualifying_orders = qualifying_orders
            self.debug_order_item3 = order_item

        print("Remaining orders with 2+ items: {:11d}".format(len(qualifying_orders)))
        print("Remaining order_item: {:21d}".format(len(order_item)))


        # Recalculate item frequency and support
        item_stats             = self.freq(order_item).to_frame("freq")
        item_stats['support']  = item_stats['freq'] / self.order_count(order_item) * 100


        # Get item pairs generator
        item_pair_gen          = self.get_item_pairs(order_item)

        if debug:
            self.debug_item_stats2 = item_stats


        # Calculate item pair frequency and support
        item_pairs              = self.freq(item_pair_gen).to_frame("freqAB")
        item_pairs['supportAB'] = item_pairs['freqAB'] / len(qualifying_orders) * 100

        print("Item pairs: {:31d}".format(len(item_pairs)))

        if debug:
            self.debug_item_pairs1 = item_pairs

        # Filter from item_pairs those below min support
        item_pairs              = item_pairs[item_pairs['supportAB'] >= min_support]

        if debug:
            self.debug_item_pairs2 = item_pairs

        print("Item pairs with support >= {}: {:10d}\n".format(min_support, len(item_pairs)))


        # Create table of association rules and compute relevant metrics
        item_pairs = item_pairs.reset_index().rename(columns={'level_0': 'item_A', 'level_1': 'item_B'})
        if debug:
            self.debug_item_pairs3 = item_pairs

        item_pairs = self.merge_item_stats(item_pairs, item_stats)
        
        item_pairs['confidenceAtoB'] = item_pairs['supportAB'] / item_pairs['supportA']
        item_pairs['confidenceBtoA'] = item_pairs['supportAB'] / item_pairs['supportB']
        item_pairs['lift']           = item_pairs['supportAB'] / (item_pairs['supportA'] * item_pairs['supportB'])
       

        # Return association rules sorted by lift in descending order

        if self.plot:
            sample_size = 50000 if assoc_pairs.shape[0]>50000 else assoc_pairs.shape[0]
            sort_size = 5000 if sample_size==50000 else int(assoc_pairs.shape[0]/10)

            fig, ax = plt.subplots(figsize=(16,12), ncols=2, nrows=2)

            sns.scatterplot(data=assoc_pairs.sample(sample_size), x='lift', y='confidenceAtoB', 
                            alpha=0.1, hue='confidenceBtoA', ax=ax[0,0])

            sns.scatterplot(data=assoc_pairs.sample(sample_size), x='lift', y='supportAB', 
                            alpha=0.1, hue='confidenceBtoA', ax=ax[0,1])

            assoc_pairs.sort_values(by='lift', ascending=False)[:sort_size].lift.plot.hist(bins=30, ax=ax[1,0])

            assoc_pairs.sort_values(by='supportAB', ascending=False)[:sort_size].supportAB.plot.hist(bins=30, ax=ax[1,1])


        return item_pairs.sort_values('lift', ascending=False)




#######################################################################################################################
# Distance Correlation: https://stats.stackexchange.com/questions/183572/understanding-distance-correlation-computations,
# https://gist.github.com/satra/aa3d19a12b74e9ab7941
# https://github.com/vnmabus/dcor
#
# Purpose: to compare different features for non-linear similarity
#######################################################################################################################

from scipy.spatial.distance import pdist, squareform
from numba import jit, float32

def _distcorr(X,Y):
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

def distcorr(Xin, Yin, per_column=True, return_df=False, columns=[]):
    """ Compute the distance correlation function
    
    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
    if per_column:
        if return_df:
            cols = Xin.columns
            if len(Yin.shape)==1:
                cols_y = 'dist_cor'
            else:
                cols_y = Yin.columns
            if len(columns)==len(cols_y):
                cols_y = columns

            Xin = Xin.values
            Yin = Yin.values
        else:
            cols = np.arange(0,Xin.shape[1])
            if "DataFrame" in str(type(X)):
                Xin = Xin.values
            if "DataFrame" in str(type(Y)):
                Yin = Yin.values

        dcor = np.zeros((Xin.shape[1], Yin.shape[1]))
        for i in range(0, Xin.shape[1]):
            for j in range(0, Yin.shape[1]):
                dcor[i, j] = _distcorr(Xin[:,i], Yin[:,j])
        if Yin.shape[1]==1:
            dcor = dcor.reshape((-1,1))

        if return_df:
            return pd.DataFrame(data=dcor, index=cols, columns=cols_y)
        else:
            return dcor
    else:
        return _distcorr(Xin, Yin)


#######################################################################################################################
# Maximal Correlation Analysis (MAC)
# http://data.bit.uni-bonn.de/publications/ICML2014.pdf
# Use MAC to find non-linearly related features, similar to distcorr and mic.
#######################################################################################################################



#######################################################################################################################
# Hausdorff distance 
#######################################################################################################################

def hausdorff(X,Y):
    '''
    Use Hausdorff to determine the set distance between e.g. the sample-set for the different targets, 
    or to find the maximally different samples between sets.    
    '''
    if "DataFrame" in str(type(X)):
        X = X.values
    if "DataFrame" in str(type(Y)):
        Y = Y.values
    # returns H distance, index of sample in X and index of sample in Y that contribute the most to the H distance 
    return  sc.spatial.distance.directed_hausdorff(X, Y)


#######################################################################################################################
# Procrustes distance 
#######################################################################################################################

def procrustes(X,Y):
    if "DataFrame" in str(type(X)):
        X = X.values
    if "DataFrame" in str(type(Y)):
        Y = Y.values

    assert X.shape[0] == Y.shape[0], "X and Y have the have the same number of rows"

    if X.shape[1] != Y.shape[1]:
        # need to add zero-columns to the matrix with less columns
        if X.shape[1] < Y.shape[1]:
            num_to_add = Y.shape[1] - X.shape[1]
            zero_cols = np.zeros((X.shape[0], num_to_add))
            X = np.hstack(X,zero_cols)
        else:
            num_to_add = X.shape[1] - Y.shape[1]
            zero_cols = np.zeros((Y.shape[0], num_to_add))
            Y = np.hstack(Y,zero_cols)
    # returns standardized version of X, orientiation of Y that best fits X and the disparity
    return sc.spatial.procrustes(X, Y)        


######################################################################################################################
# Maximal Information Coefficient: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3325791/
#
# Purpose: to compare different feature for non-linear similarity
#######################################################################################################################

from minepy import MINE
from minepy import cstats as MI_cstats, pstats as MI_pstats

def mic_scores(X, Y=None, alpha=0.6, c=16, est='mic_e', return_df=False):
    mic_miner = MINE(alpha=alpha, c=c, est=est)

    if "DataFrame" in str(type(X)):
        cols = X.columns
        X=X.values

    mic_results = np.zeros((X.shape[1], 3))
    for j in range(0, X.shape[1]):
        mic_miner.compute_score(X[:, j], Y)
        mic_results[j,:] = mic_miner.mic(), mic_miner.tic(norm=True), mic_miner.mas()

    if return_df:
        return pd.DataFrame(data=mic_results, index=cols, columns=['MIC', 'TIC', 'MAS'])
    else:
        return mic_results



#######################################################################################################################
# Bayes Factor
# the sampling-brother of the likelihood ratio test
#######################################################################################################################



#######################################################################################################################
# Random Matrix Theory
#######################################################################################################################




#######################################################################################################################
# Odds ratio
#######################################################################################################################




#######################################################################################################################
# Likelihood ratio test
#######################################################################################################################

from scipy.stats import power_divergence as pdiv


def powerdiv_scores(X,y, bins=10, pdivtype='llr'):
    if pdivtype=='llr':
        _lambda = 0
    elif pdivtype=='mllr':
        _lambda = -1
    elif pdivtype=='neyman':
        _lambda = -2
    elif pdivtype=='pearson':
        _lambda = 1
    elif pdivtype=='cressie-read':
        _lambda = 2/3


    if "DataFrame" in str(type(X)):
        X = X.values
    scores = np.zeros((X.shape[1],))
    for jdx in range(0, scores.shape[0]):
        xa = X[:, jdx]
        x1 = xa[np.argwhere(y == 0)]
        x2 = xa[np.argwhere(y == 1)]
        qranges = np.quantile(xa, np.arange(0, 1, 1/bins))
        scores[jdx] = powerdiv_score(x1, x2, qranges, bins, _lambda)
    return scores

def powerdiv_score(x1, x2, qranges, bins, _lambda):
    freq1, freq2 = np.zeros((bins,), dtype=int), np.zeros((bins,), dtype=int)
    for i in range(0, bins-1):
        freq1[i] = np.where((x1 >= qranges[i]) & (x1 < qranges[i+1]))[0].shape[0]
        freq2[i] = np.where((x2 >= qranges[i]) & (x2 < qranges[i+1]))[0].shape[0]
    freq1[i+1] = np.where(x1 >= qranges[i+1])[0].shape[0]
    freq2[i+1] = np.where(x2 >= qranges[i+1])[0].shape[0]
    return pdiv(freq1, freq2, lambda_ = _lambda)[0]

#######################################################################################################################
#######################################################################################################################

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

class ReduceVIF(BaseEstimator, TransformerMixin):
    # https://www.kaggle.com/ffisegydd/sklearn-multicollinearity-class
    def __init__(self, thresh=5.0, impute=True, impute_strategy='median'):
        # From looking at documentation, values between 5 and 10 are "okay".
        # Above 10 is too high and so should be removed.
        self.thresh = thresh

        # The statsmodel function will fail with NaN values, as such we have to impute them.
        # By default we impute using the median value.
        # This imputation could be taken out and added as part of an sklearn Pipeline.
        if impute:
            if impute_strategy in ['median', 'mean', 'most_frequent']:
                self.imputer = SimpleImputer(strategy=impute_strategy)
            elif impute_strategy=='iterative':
                self.imputer = IterativeImputer()
            else:
                raise(ValueError, f"imputance: {impute_strategy} is not supported at this moment")

    def fit(self, X, y=None):
        print('ReduceVIF fit')
        if hasattr(self, 'imputer'):
            self.imputer.fit(X)
        return self

    def transform(self, X, y=None):
        print('ReduceVIF transform')
        columns = X.columns.tolist()
        if hasattr(self, 'imputer'):
            X = pd.DataFrame(self.imputer.transform(X), columns=columns)
        return ReduceVIF.calculate_vif(X, self.thresh)

    @staticmethod
    def calculate_vif(X, thresh=5.0):
        # Taken from https://stats.stackexchange.com/a/253620/53565 and modified
        dropped = True
        while dropped:
            variables = X.columns
            dropped = False
            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]

            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                dropped = True
        return X












#######################################################################################################################
#######################################################################################################################

def get_accuracy_plots(y_test, y_pred, figax=None, make_plot=True):
    #
    threshold = np.arange(0., 1, 0.025)
    _metrics = []
    for _thresh in threshold:
        bal_acc = balanced_accuracy(y_test, y_pred, thresh=_thresh)
        f1_score = fb_score(y_test, y_pred, beta=1, thresh=_thresh)
        _npv = npv(y_test, y_pred, thresh=_thresh)
        _fpr = fpr(y_test, y_pred, thresh=_thresh)
        rec, true_rec = recall(y_test, y_pred, thresh=_thresh)
        _metrics.append({'BAL_ACC': bal_acc,
                         'F1_SCORE': f1_score,
                         'NPV': _npv,
                         'REC': rec,
                         'TRUE_REC': true_rec,
                         'FPR': _fpr,
                         'AUC': metrics.roc_auc_score(y_test, y_pred),
                         'THRESHOLD': _thresh})
    _metrics = pd.DataFrame(_metrics)

    if make_plot:
        if figax is None:
            fig, ax = plt.subplots(ncols=4, figsize=(28, 8))
        else:
            fig, ax = figax
        pd.DataFrame(y_pred).hist(bins=20, ax=ax[1], histtype='step')
        pd.DataFrame(y_test).hist(bins=2, ax=ax[1], color='black', histtype='step')
        ax[1].set_title('Proba histo')

        # sns.lineplot(data=_metrics, x='THRESHOLD', y='F1_SCORE', label='F1', ax=ax[1])
        sns.lineplot(data=_metrics, x='NPV', y='REC', color='green', ax=ax[0], ci=None)
        sns.lineplot(data=_metrics, x='NPV', y='TRUE_REC', color='red', ax=ax[0], ci=None)
        # sns.lineplot(data=_metrics, x='THRESHOLD', y='TRUE_REC', label='TRUE_REC', ax=ax[1])
        # sns.lineplot(data=_metrics, x='THRESHOLD', y='BAL_ACC', label='BAL_ACC', ax=ax[1])
        ax[0].set_title('NPV versus recall')
        ax[0].set_ylabel("recall")

        # TPR / FPR -> sensitivity / 1-specifity
        roc_curve = pd.DataFrame(metrics.roc_curve(y_test, y_pred)[:2]).transpose()
        roc_curve.columns = ['FPR', 'TPR']
        sns.lineplot(data=roc_curve, x='FPR', y='TPR', ax=ax[2], ci=None)
        # sns.scatterplot(data=roc_curve, x='FPR', y='TPR', ax=ax[2],s=100)
        # ax[2].plot(np.arange(0,1,0.05),np.arange(0,1,0.05), color='black', )
        ax[2].plot(np.array([0, 1]), np.array([0, 1]), ls="--", c="black")
        ax[2].set_title("ROC")

        prec_recall = pd.DataFrame(metrics.precision_recall_curve(y_test, y_pred)[:2]).transpose()
        prec_recall.columns = ['precision', 'recall']
        sns.lineplot(data=prec_recall, x='precision', y='recall', ax=ax[3], ci=None)
        ax[3].set_title("precision - recall")
    else:
        fig, ax = None, None

    return _metrics, (fig, ax)


def mcc(y_true, y_prob, thresh=0.5, greedy='symmetric'):
    if greedy == 'symmetric':
        TP = np.sum((y_prob > thresh) & (y_true == 1))
        TN = np.sum((y_prob < (1 - thresh)) & (y_true == 0))
        FP = np.sum((y_prob > thresh) & (y_true == 0))
        FN = np.sum((y_prob < (1 - thresh)) & (y_true == 1))
    elif greedy == 'negative':
        TP = np.sum((y_prob > thresh) & (y_true == 1))
        TN = np.sum((y_prob < (thresh)) & (y_true == 0))
        FP = np.sum((y_prob > thresh) & (y_true == 0))
        FN = np.sum((y_prob < (thresh)) & (y_true == 1))
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) if (TP + FP) * (TP + FN) * (
                TN + FP) * (TN + FN) > 0 else np.nan
    return mcc


def balanced_accuracy(y_true, y_prob, thresh=0.5, greedy='symmetric'):
    if greedy == 'symmetric':
        TP = np.sum((y_prob > thresh) & (y_true == 1))
        TN = np.sum((y_prob < (1 - thresh)) & (y_true == 0))
        FP = np.sum((y_prob > thresh) & (y_true == 0))
        FN = np.sum((y_prob < (1 - thresh)) & (y_true == 1))
    elif greedy == 'negative':
        TP = np.sum((y_prob > thresh) & (y_true == 1))
        TN = np.sum((y_prob < (thresh)) & (y_true == 0))
        FP = np.sum((y_prob > thresh) & (y_true == 0))
        FN = np.sum((y_prob < (thresh)) & (y_true == 1))

    recall = TP / (FN + TP) if (FN + TP) > 0 else np.nan
    specificity = TN / (FP + TN) if (FP + TN) > 0 else np.nan
    return 0.5 * (recall + specificity)


def fb_score(y_true, y_prob, beta=1, thresh=0.5, greedy='symmetric'):
    if greedy == 'symmetric':
        TP = np.sum((y_prob > thresh) & (y_true == 1))
        TN = np.sum((y_prob < (1 - thresh)) & (y_true == 0))
        FP = np.sum((y_prob > thresh) & (y_true == 0))
        FN = np.sum((y_prob < (1 - thresh)) & (y_true == 1))
    elif greedy == 'negative':
        TP = np.sum((y_prob > thresh) & (y_true == 1))
        TN = np.sum((y_prob < (thresh)) & (y_true == 0))
        FP = np.sum((y_prob > thresh) & (y_true == 0))
        FN = np.sum((y_prob < (thresh)) & (y_true == 1))
    prec = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    rec = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    return (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec)


def npv(y_true, y_prob, thresh=0.5, greedy='symmetric'):
    if greedy == 'symmetric':
        TN = np.sum((y_prob < (1 - thresh)) & (y_true == 0))
        FN = np.sum((y_prob < (1 - thresh)) & (y_true == 1))
    elif greedy == 'negative':
        TN = np.sum((y_prob < (thresh)) & (y_true == 0))
        FN = np.sum((y_prob < (thresh)) & (y_true == 1))
    NPV = TN / (TN + FN) if (TN + FN) > 0 else np.nan
    return NPV


def recall(y_true, y_prob, thresh=0.5, greedy='symmetric'):
    if greedy == 'symmetric':
        TP = np.sum((y_prob > thresh) & (y_true == 1))
        FN = np.sum((y_prob < (1 - thresh)) & (y_true == 1))
    elif greedy == 'negative':
        TP = np.sum((y_prob > thresh) & (y_true == 1))
        FN = np.sum((y_prob < (thresh)) & (y_true == 1))
    AP = np.sum(y_true)
    recall = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    return recall, TP / AP


def prec(y_true, y_prob, thresh=0.5, greedy='symmetric'):
    TP = np.sum((y_prob > thresh) & (y_true == 1))
    FP = np.sum((y_prob > thresh) & (y_true == 0))
    prec = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    return prec


def spec(y_true, y_prob, thresh=0.5, greedy='symmetric'):
    TN = np.sum((y_prob < (1 - thresh)) & (y_true == 0))
    FP = np.sum((y_prob > thresh) & (y_true == 0))
    spec = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    return spec


def fpr(y_true, y_prob, thresh=0.5, greedy='symmetric'):
    if greedy == 'symmetric':
        TN = np.sum((y_prob < (1 - thresh)) & (y_true == 0))
        FP = np.sum((y_prob > thresh) & (y_true == 0))
    elif greedy == 'negative':
        TN = np.sum((y_prob < (thresh)) & (y_true == 0))
        FP = np.sum((y_prob > thresh) & (y_true == 0))
    FPR = FP / (FP + TN) if (FP + TN) > 0 else np.nan
    return FPR


def roc(y_true, y_prob, thresh):
    roc_arr = []
    for _thresh in thresh:
        TP = np.sum((y_prob > thresh) & (y_true == 1))
        TN = np.sum((y_prob < (1 - thresh)) & (y_true == 0))
        FP = np.sum((y_prob > thresh) & (y_true == 0))
        FN = np.sum((y_prob < (1 - thresh)) & (y_true == 1))

        TPR = TP / (TP + FN) if (TP + FN) > 0 else np.nan
        FPR = FP / (FP + TN) if (FP + TN) > 0 else np.nan
        roc_arr.append((TPR, FPR))
    return roc_arr

def precr(y_true, y_prob, thresh):
    precr_arr = []
    for _thresh in thresh:
        TP = np.sum((y_prob > thresh) & (y_true == 1))
        TN = np.sum((y_prob < (1 - thresh)) & (y_true == 0))
        FP = np.sum((y_prob > thresh) & (y_true == 0))
        FN = np.sum((y_prob < (1 - thresh)) & (y_true == 1))

        REC = TP / (FN + TP) if (FN + TP) > 0 else np.nan
        PREC = TP / (TP + FP) if (TP + FP) > 0 else np.nan
        precr_arr.append((PREC, REC))
    return prec_arr
'''
class feature_expansion():
    def __init__():
        return True
    def bulk_feature_expander():
        return True
    def iterative_feature_expander():
        return True
class clusterizer():
    def __init__():
        return True
    def _transformer():
        return True
    def _filter():
        return True
    def _reducer():
        return True
    def _cluster():
        return True


    def cluster_separation():
        return True 
class classifier():
    def __init__():
        return True

    def _transformer():
        return True


    def _filter():
        return True

    def _reducer():
        return True
        

    def _classifier():
        return True

    def classifier_scoring():
        return True
'''