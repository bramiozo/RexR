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

from scipy.stats import ks_2samp as ks, wasserstein_distance as wass, spearmanr, energy_distance
from scipy.stats import chisquare, epps_singleton_2samp as epps

from sklearn.decomposition import PCA, FastICA as ICA, FactorAnalysis as FA
from sklearn.feature_selection import mutual_info_classif as Minfo, f_classif as Fval, chi2


'''
class binomial_feature_evaluator


class multinomial_feature_evaluator


class regression_feature_evaluator


class multi_omic_evaluator
    feature_expander



'''


def get_statdist_dataframe_binomial(X,Y, features):
    assert features is not None, 'You are required to provide the feature list,\n\r\t\t otherwise the dataframe cannot be indexed'
    assert len(features)== X.shape[1], 'The number of feature names given is not equal to the number of columns'
    assert len(Y) == X.shape[0], 'The number of rows in the matrix X is not the equal to the length of the target vector Y'

    stat_dist = dict()
    stat_dist['diffentropy'] = diff_entropy_scores(X, eps=1e-6, bins=30)
    stat_dist['variance'] = variance_scores(X)
    stat_dist['pca_imp'] = get_reducer_weights(X, PCA(n_components=50), cols=None, ncomp=50)
    stat_dist['fa_imp'] = get_reducer_weights(X, FA(n_components=50), cols=None, ncomp=50)
    stat_dist['ica_imp'] = get_reducer_weights(X, ICA(n_components=50), cols=None, ncomp=50)

    stat_dist['minf'] = Minfo(X, Y)
    stat_dist['fscore'] = Fval(X, Y)
    stat_dist['Wass1'] = wass1_scores(X, Y)

    fswass2 = fs_ws2()
    stat_dist['Wass2'] = fswass2.fit(X, Y).scores_

    stat_dist['spearman'] = spearman_scores(X, Y)
    stat_dist['ks'] = ks_scores(X, Y)

    stat_dist['seqentropy'] = seq_entropy_scores(X, Y)
    stat_dist['qseqentropy_prod'] = qseq_entropy_scores(X, Y, q_type='prod', bins=10)
    stat_dist['qseqentropy_sum'] = qseq_entropy_scores(X, Y, q_type='sum', bins=10)
    stat_dist['seqentropyX'] = seqX_entropy_scores(X, Y, seqrange=(5, 25))

    stat_dist['cdf_1'] = cdf_scoresB(X,Y, dist_type='mink_rao')
    stat_dist['cdf_2'] = cdf_scoresB(X,Y, dist_type='mink_rao2')
    stat_dist['cdf_3'] = cdf_scoresB(X,Y, dist_type='rao')
    stat_dist['cdf_4'] = cdf_scoresG(X,Y, dist_type='emd')
    stat_dist['cdf_5'] = cdf_scoresG(X,Y, dist_type='cvm')
    stat_dist['cdf_6'] = cdf_scoresG(X,Y, dist_type='cust')

    stat_dist['med_dist'] = q_dists(X,Y, q=0.5)
    stat_dist['q25_dist'] = q_dists(X,Y, q=0.25) 
    stat_dist['q75_dist'] = q_dists(X,Y, q=0.75)
    stat_dist['var_dist'] = var_dists(X,Y)
    stat_dist['q5_acc'] = q_acc_scores(X,Y, q=0.5)
    stat_dist['q75_acc'] = q_acc_scores(X,Y, q=0.75)

    stat_dist['KL'] = ec_scores(X,Y, num_bins=7, ent_type='kl')
    stat_dist['Shan'] = ec_scores(X,Y, num_bins=7, ent_type='shannon')
    stat_dist['Cross'] = ec_scores(X,Y, num_bins=7, ent_type='cross')

    fsepps = fs_epps(pvalue=0.01)
    stat_dist['Chi2'] = chi2_scores(X, Y, bins=7)
    stat_dist['epps'] = fsepps.fit(X, Y).scores_


    # Combine in dataframe
    stat_dist_df = pd.DataFrame(data=np.vstack([stat_dist['diffentropy'],
                                                stat_dist['variance'],
                                                stat_dist['pca_imp'],
                                                stat_dist['fa_imp'],
                                                stat_dist['ica_imp'],
                                                stat_dist['minf'], 
                                                stat_dist['fscore'], 
                                                stat_dist['Wass1'],
                                                stat_dist['Wass2'],
                                                stat_dist['KL'],
                                                stat_dist['spearman'].T, 
                                                stat_dist['ks'].T,
                                                stat_dist['Shan'],
                                                stat_dist['Cross'],
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
                                                stat_dist['Chi2'],
                                                stat_dist['epps']]).T,
                               columns=['diffentropy', 'variance',
                                        'pca_imp', 'fa_imp', 'ica_imp',
                                        'Minf', 
                                        'Fscore', 'FPval', 
                                        'Wass1', 'Wass2', 'KL',
                                        'SpearmanScore', 'SpearmanPval',
                                        'KSScore', 'KSPval', 'Shannon', 'Cross',
                                        'seqentropy_wass1', 'qseqentropy_prod_wass1', 'qseqentropy_sum_wass1', 'seqentropyX_wass1',
                                        'CDF1', 'CDF2', 'CDF3', 'CDF4', 'CDF5', 'CDF6',
                                        'q5delta', 'q25delta', 'q75delta', 'var_dist', 'q5_acc', 'q75_acc', 
                                        'Chi2', 'Epps'],
                               index=features)
    return stat_dist_df    


   
def get_reducer_weights(X, reducer=None, cols=None, ncomp=10):
    '''
     reducer: e.g PCA, FA, ICA
    '''
    reducer.fit(X)    
    pcweights = np.zeros((X.shape[1],))
    wt = 0
    for pc in range(0, ncomp):
        w = ncomp-pc
        wt = wt + w
        pcweights = w*np.abs(reducer.components_[pc]) + pcweights
    pcweights = pcweights/wt    
    if cols is not None:
        pcw = pcweights[np.argsort(pcweights)]
        _cols = np.array(cols)[np.argsort(pcweights)]
        return dict(zip(_cols,pcw))
    else:
        return pcweights



@jit
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

@jit
def diff_entropy_scores(X, eps=1e-6, bins=20):
    if "DataFrame" in str(type(X)):
        X = X.values
    scores = np.zeros((X.shape[1],))
    for jdx in range(0, scores.shape[0]):
        xa = X[:, jdx]
        scores[jdx] = _diff_entropy(xa, eps=eps, bins=bins)
    return scores

@jit
def _diff_entropy(x, eps=1e-6, bins=20):
    rhos, xs = np.histogram(x, density=True, bins=bins)
    xdiff = xs[1:] - xs[:-1]
    H = np.sum(rhos*np.log(rhos+eps)*xdiff)
    Hr = H/np.sum(xdiff)
    return Hr

@jit
def variance_scores(X):
    if "DataFrame" in str(type(X)):
        X = X.values
    scores = np.zeros((X.shape[1],))
    for jdx in range(0, scores.shape[0]):
        xa = X[:, jdx]
        scores[jdx] = np.var(xa)
    return scores

@jit
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
        scores[jdx] = np.max([np.mean(yl)-cr, np.mean(ys)-cr])/cr
    return scores

@jit
def chi2_scores(X,y, bins=10):
    if "DataFrame" in str(type(X)):
        X = X.values
    scores = np.zeros((X.shape[1],))
    for jdx in range(0, scores.shape[0]):
        xa = X[:, jdx]
        x1 = xa[np.argwhere(y == 0)]
        x2 = xa[np.argwhere(y == 1)]
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
@jit
def cdf_scoresB(X, y, dist_type="mink_rao"):
    if "DataFrame" in str(type(X)):
        X = X.values
    scores = np.zeros((X.shape[1],))
    for jdx in range(0, scores.shape[0]):
        scores[jdx] = _cdf_distanceB(X[np.argwhere(y==0)[:,0], jdx],
                                     X[np.argwhere(y==1)[:,0], jdx],
                                     bin_size=5,
                                     minkowski=1,
                                     dist_type=dist_type)
    return scores

@jit
def cdf_scoresG(X, y, dist_type="emd"):
    if "DataFrame" in str(type(X)):
        X = X.values
    scores = np.zeros((X.shape[1],))
    for jdx in range(0, scores.shape[0]):
        scores[jdx] = _cdf_distanceG(X[np.argwhere(y==0)[:,0], jdx],
                                     X[np.argwhere(y==1)[:,0], jdx],
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


def _cdf_distanceG(x1, x2, bin_size=15, dist_type='emd'):
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
        return np.sum(lt1[:, 0] * (np.abs(l2 - l1))) / c  # qEMD
    elif dist_type == 'cvm':
        return np.sqrt(np.sum(lt1[:, 0] * np.power((l2 - l1), 2))) / c  # qCvM
    elif dist_type == 'cust':
        return np.sum(lt1[:, 0] * (l2 - l1)) / c

"""
Ansatz for sequence entropy
"""

@jit
def seq_entropy_scores(X, y):
    if "DataFrame" in str(type(X)):
        X = X.values
    C = np.unique(y).shape[0]

    scores = np.zeros((X.shape[1],))
    for jdx in range(0, X.shape[1]):
        y_sorted = y[np.argsort(X[:, jdx])]
        scores[jdx] = seq_entropy(y_sorted, C)
    return scores

@jit
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


@jit
def qseq_entropy_scores(X, y, q_type='sum', bins=20):
    if "DataFrame" in str(type(X)):
        X = X.values
    C = np.unique(y).shape[0]

    scores = np.zeros((X.shape[1],))
    for jdx in range(0, X.shape[1]):
        y_sorted = y[np.argsort(X[:, jdx])]
        scores[jdx] = qseq_entropy(y_sorted, q_type=q_type, bins=bins)
    return scores


@jit
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


@jit
def seq_prob(n, k, p):
    div = sc.special.factorial(n) / sc.special.factorial(k) / sc.special.factorial(n - k)
    probs = np.power(p, k) * np.power(1 - p, n - k)
    return div * probs * k


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
def seqX_entropy_scores(X, y, seqrange=(2, 20)):
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

"""
Variance distances
"""
@jit
def var_dists(X, y):
    if "DataFrame" in str(type(X)):
        X = X.values
    scores = np.zeros((X.shape[1],))
    for jdx in range(0, scores.shape[0]):
        #gvar = np.var(X[:, jdx])
        scores[jdx] = var_dist(X[np.argwhere(y==0)[:, 0], jdx],
                           X[np.argwhere(y==1)[:, 0], jdx])
    return scores

@jit
def var_dist(x1, x2):
    return (np.var(x2)-np.var(x1))

"""
Quantile distances
"""

@jit
def q_dists(X, y, q=0.5):
    if "DataFrame" in str(type(X)):
        X = X.values
    scores = np.zeros((X.shape[1],))
    for jdx in range(0, scores.shape[0]):
        scores[jdx] = q_dist(X[np.argwhere(y==0)[:,0], jdx],
                           X[np.argwhere(y==1)[:,0], jdx], q=q)
    return scores

@jit
def q_dist(x1, x2, q=0.5, weighted=True):
    q1 = np.quantile(x1, q)
    q2 = np.quantile(x2, q)
    if weighted:
        q1std = np.std(x1)
        q2std = np.std(x1)
        return (q2 - q1)/np.min([q1std, q2std])
    else:
        return q2 - q1

"""
Spearman applied to array
"""

@jit
def spearman_scores(X, y):
    if "DataFrame" in str(type(X)):
        X = X.values
    C = np.unique(y).shape[0]

    scores = np.zeros((X.shape[1], 2))
    for jdx in range(0, X.shape[1]):
        sorted_ind = np.argsort(X[:, jdx])
        x_sorted = X[sorted_ind, jdx]
        y_sorted = y[sorted_ind]
        scores[jdx, :] = spearmanr(x_sorted, y_sorted)
    return scores


"""
Kolmogorov-Smirnov applied to array
"""

@jit
def ks_scores(X,y):
    if "DataFrame" in str(type(X)):
        X = X.values
    scores = np.zeros((X.shape[1], 2))
    for jdx in range(0, scores.shape[0]):
        scores[jdx,:] = ks(X[np.argwhere(y==0)[:,0], jdx], 
                           X[np.argwhere(y==1)[:,0], jdx])
    return scores



"""
Wasserstein 1 distance applied to array
"""
@jit
def wass1_scores(X,y):
    if "DataFrame" in str(type(X)):
        X = X.values
    scores = np.zeros((X.shape[1],))
    for jdx in range(0, scores.shape[0]):
        scores[jdx] = wass(X[np.argwhere(y==0)[:,0], jdx], 
                           X[np.argwhere(y==1)[:,0], jdx])
    return scores


"""
KL divergence applied to array
"""
@jit
def ec_scores(X,y, num_bins=25, ent_type='kl'):
    if "DataFrame" in str(type(X)):
        X = X.values
    scores = np.zeros((X.shape[1],))
    for jdx in range(0, scores.shape[0]):
        scores[jdx] = _information_change(X[np.argwhere(y==0)[:,0], jdx], 
                           		  X[np.argwhere(y==1)[:,0], jdx], num_bins=num_bins, ent_type='kl')
    return scores   


"""
     Information change: 
        * Kullback-Leibler divergence (have to make same number of bins)
        * cross-entropy
        * Shannon entropy change
"""
@jit
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
        pos_samples = x[one_idx]
        neg_samples = x[zero_idx]
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


def _cov(x, lib='numpy', method='exact', inverse=False):
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
            invconv = cm.precision_
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
    return



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
def _wcorr(x1, x2, w):
    '''
    x1, x2 : array of size (N samples,) or (N, m)
    weights : weight of each sample/feature
    returns the weighted Pearson correlation
    '''     
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

def _boolean_sim(x1, x2, fun=None, w=None):
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