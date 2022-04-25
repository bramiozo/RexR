from __future__ import division
import numpy as np
import itertools
from numba import jit, njit
from numpy.core.fromnumeric import var
import torch
import scipy as sc
from scipy.stats import ks_2samp as ks, wasserstein_distance as wass, spearmanr
from scipy.stats import energy_distance, pearsonr, kendalltau, theilslopes, weightedtau
from scipy.stats import chisquare, epps_singleton_2samp as epps
from scipy.stats import power_divergence as pdiv

from sklearn.metrics import mutual_info_score
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels

import numpy as np
from scipy.stats import gamma
from scipy.stats import multiscale_graphcorr


######################################################################################################################
# Global correlation coefficient: g = np.sqrt(1-np.inverse(V_kk*np.inverse(Cov)_kk))
######################################################################################################################
def global_corr(X,c=None, sparse=False):
    cov, invcov = _cov(X, inverse=True)   
    if c is not None:
        # only consider column c
        gunit = np.zeros((1,))
        gunit[0] = np.sqrt(1-1/(invcov[c,c])/cov[[c,c]])
    else:
        # get global correlation for all columns
        gunit = np.zeros((X.shape[1],))
        for c in range(X.shape[1]):
            gunit[c] = np.sqrt(1-1/(invcov[c,c])/cov[[c,c]])
    return gunit


######################################################################################################################
# Goodman-Kruskall's Gamma
# Rank correlation, paired
######################################################################################################################

@jit
def _get_pairs(v1, v2):
    K = int(len(v1)**2-len(v1)/2)
    k = 0
    pairs = np.zeros((K,4))
    for i in range(len(v1)):
        for j in range(i+1, len(v1)):
            pairs[k, :] = np.hstack([v1[i], v2[i], v1[j], v2[j]])
            k += 1
    return pairs

def _get_concordant_pairs(v1, v2):
    _pairs = _get_pairs(v1,v2)
    return sum((np.sign(_pairs[:,0]-_pairs[:,2]) == 
                    np.sign(_pairs[:,1]-_pairs[:,3])).astype(np.int32))

def _get_discordant_pairs(v1, v2):
    _pairs = _get_pairs(v1,v2)
    return sum((np.sign(_pairs[:,0]-_pairs[:,2]) != 
                    np.sign(_pairs[:,1]-_pairs[:,3])).astype(np.int32))
    
def goodman_kruskal_gamma(v1, v2):
    assert type(v1) == np.ndarray, 'v1 is not a numpy array'
    assert type(v2) == np.ndarray, 'v2 is not a numpy array'
    assert v1.shape == v2.shape, 'Shapes of v1, v2 should be the same'
    assert len(v1.shape)>=1, 'v1 is not an Nd array'
    assert len(v2.shape)>=1, 'v2 is not an Nd array'
    
    # G = (Nc - Nd)/(Nc+Nd)
    # Nc = # of concordant pairs
    # Nd = # of discordant pairs
    # pairs are concordant, discordant or tied
    K = int(len(v1)**2//2-len(v1))
    Nc = _get_concordant_pairs(v1, v2)
    Nd = _get_discordant_pairs(v1, v2)   
    return (Nc-Nd)/(Nc+Nd)


######################################################################################################################
# Cramer K-ish
# create empirical bi-variate distribution of two features and determine Chi2 relative to expected Chi2 if it were
# a bi-variate normal distribution. This is part of the https://github.com/KaveIO/PhiK package
# https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
# https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
######################################################################################################################

from scipy.stats import mvn

def _mvn_un(rho: float, lower: tuple, upper: tuple) -> float:
    '''Perform integral of bivariate normal gauss with correlation
    Integral is performed using scipy's mvn library.
    :param float rho: tilt parameter
    :param tuple lower: tuple of lower corner of integral area
    :param tuple upper: tuple of upper corner of integral area
    :returns float: integral value
    '''
    mu = np.array([0., 0.])
    S = np.array([[1., rho], [rho, 1.]])
    p, i = mvn.mvnun(lower, upper, mu, S)
    return p

def _mvn_array(rho: float, sx: np.ndarray, sy: np.ndarray) -> list:
    '''Array of integrals over bivariate normal gauss with correlation
    Integrals are performed using scipy's mvn library.
    
    :param float rho: tilt parameter
    :param np.ndarray sx: bin edges array of x-axis
    :param np.ndarray sy: bin edges array of y-axis
    :returns list: list of integral values
    '''
    # ranges = [([sx[i], sy[j]], [sx[i+1], sy[j+1]]) for i in range(len(sx) - 1) for j in range(len(sy) - 1)]
    # corr = [mvn.mvnun(lower, upper, mu, S)[0] for lower, upper in ranges]
    # return corr

    # mean and covariance
    mu = np.array([0., 0.])
    S = np.array([[1., rho], [rho, 1.]])

    # callling mvn.mvnun is expansive, so we only calculate half of the matrix, then symmetrize
    # add half block, which is symmetric in x
    odd_odd = False
    ranges = [([sx[i], sy[j]], [sx[i+1], sy[j+1]]) for i in range((len(sx)-1)//2) for j in range(len(sy) - 1)]
    # add odd middle row, which is symmetric in y
    if (len(sx)-1) % 2 == 1:
        i = (len(sx)-1)//2
        ranges += [([sx[i], sy[j]], [sx[i+1], sy[j+1]]) for j in range((len(sy)-1)//2)]
        # add center point, add this only once
        if (len(sy)-1) % 2 == 1:
            j = (len(sy)-1)//2
            ranges.append(([sx[i], sy[j]], [sx[i+1], sy[j+1]]))
            odd_odd = True

    corr = np.array([_calc_mvnun(lower, upper, mu, S) for lower, upper in ranges])
    # add second half, exclude center
    corr = np.concatenate([corr, corr if not odd_odd else corr[:-1]])
    return corr

def _calc_mvnun(lower, upper, mu, S):
    return mvn.mvnun(lower, upper, mu, S)[0]

def _chi2_from_phik(rho: float, n: int, subtract_from_chi2:float=0,
                   corr0:list=None, scale:float=None, sx:np.ndarray=None, sy:np.ndarray=None,
                   pedestal:float=0, nx:int=-1, ny:int=-1) -> float:
    '''Calculate chi2-value of bivariate gauss having correlation value rho
    
        Calculate no-noise chi2 value of bivar gauss with correlation rho,
        with respect to bivariate gauss without any correlation.
        
        :param float rho: tilt parameter
        :param int n: number of records
        :param float subtract_from_chi2: value subtracted from chi2 calculation. default is 0.
        :param list corr0: mvn_array result for rho=0. Default is None.
        :param float scale: scale is multiplied with the chi2 if set.
        :param np.ndarray sx: bin edges array of x-axis. default is None.
        :param np.ndarray sy: bin edges array of y-axis. default is None.
        :param float pedestal: pedestal is added to the chi2 if set.
        :param int nx: number of uniform bins on x-axis. alternative to sx.
        :param int ny: number of uniform bins on y-axis. alternative to sy.
        :returns float: chi2 value
    '''

    if corr0 is None:
        corr0 = _mvn_array(0, sx, sy)
    if scale is None:
        # scale ensures that for rho=1, chi2 is the maximum possible value
        corr1 = _mvn_array(1, sx, sy)
        delta_corr2 = (corr1 - corr0) ** 2
        # protect against division by zero
        ratio = np.divide(delta_corr2, corr0, out=np.zeros_like(delta_corr2), where=corr0!=0)
        chi2_one = n * np.sum(ratio)
        # chi2_one = n * sum([((c1-c0)*(c1-c0)) / c0 for c0, c1 in zip(corr0, corr1)])
        chi2_max = n * min(nx-1, ny-1)
        scale = (chi2_max - pedestal) / chi2_one

    corrr = _mvn_array(rho, sx, sy)
    delta_corr2 = (corrr - corr0) ** 2
    # protect against division by zero
    ratio = np.divide(delta_corr2, corr0, out=np.zeros_like(delta_corr2), where=corr0!=0)
    chi2_rho = n * np.sum(ratio)
    # chi2_rho = (n * sum([((cr-c0)*(cr-c0)) / c0 for c0, cr in zip(corr0, corrr)]))

    chi2 = pedestal + chi2_rho * scale
    return chi2 - subtract_from_chi2

def cramer_kish(v1,v2, nbins=5, nruns=100, fitness_test=pdiv, c=0, **kwargs):
    # fitness_test : chi2_contingency, power_divergence, fisher_exact
    # lambda_:0 for llr, lambda_:1 for chi2, lambda_:-1 for mllr
    N=len(v1)
    pref = 1
    ecov = np.cov(v1, v2)
    emp_m = np.array([np.median(v1), np.median(v2)], axis=1)

    x2_list = []
    p2_list = []
    #for k in range(nruns):
    bivar=np.random.multivariate_normal(emp_m, ecov, size=N)
    bivar_freq = pref*np.histogram2d(bivar[:,0], bivar[:,1], bins=nbins, density=False)[0]

    emp_freq, sx, sy = pref*np.histogram2d(v1, v2, bins=nbins, density=False)
    #for i in range(nbins):
    #    for j in range(nbins):
    #        bivar_freq[i,j]= 1//N*(np.sum(emp_freq[i,:])*np.sum(emp_freq[:,j]))

    bivar_freq = bivar_freq
    emp_freq = emp_freq

    nsdof = (nbins-1)**2 - np.sum(bivar_freq==0) 
    x2ped = nsdof + c*np.sqrt(2.* nsdof)
    x2max = N*(nbins-1)
    
    x2, p2, _, _ = chi2_contingency(emp_freq, lambda_='log-likelihood')

    if x2<x2ped:
        return 0, p2
    elif x2>=x2max:
        return 1, 0
    else:
        corr0 = _mvn_array(0, sx, sy)
        corr1 = _mvn_array(1, sx, sy)
        delta_corr2 = (corr1 - corr0) ** 2
        ratio = np.divide(delta_corr2, corr0, out=np.zeros_like(delta_corr2), where=corr0!=0)
        x2_one = N * np.sum(ratio)
        scale = (x2max - x2ped) / x2_one
        return brentq(_chi2_from_phik, 0, 1, args=(N, chi2, corr0, scale, sx, sy, x2ped), xtol=1e-5), p2

    #x2 = fitness_test(bivar_freq, emp_freq, **kwargs)[0]
    #p2 = fitness_test(bivar_freq, emp_freq, **kwargs)[1]

    # chi2ped = nbins**2
    # chi2max = N*nbins*pref
    # x2 = chi2ped + x2*(chi2max-chi2ped)

    return 


def cramer_phi(v1,v2, nbins=5, nruns=100, fitness_test=pdiv, **kwargs):
    # fitness_test : chi2_contingency, power_divergence, fisher_exact
    # lambda_:0 for llr, lambda_:1 for chi2, lambda_:-1 for mllr
    N=len(v1)
    pref = 1

    emp_freq = pref*np.histogram2d(v1, v2, bins=nbins, density=False)[0]
    x2s = np.zeros((nbins, nbins))
    for i in range(nbins):
        for j in range(nbins):
            x2s[i,j] = np.square(emp_freq[i,j] -  1//N*(np.sum(emp_freq[i,:])*np.sum(emp_freq[:,j])))
            x2s[i,j]/= 1//N*(np.sum(emp_freq[i,:])*np.sum(emp_freq[:,j]))

    x2 = np.sum(x2s[i,j])

    # get p-val
    coincnt = 0
    for nrun in range(nruns):
        emp_freq = pref*np.histogram2d(v1[np.random.randint(low=0, high=N, size=N)], 
                                       v2[np.random.randint(low=0, high=N, size=N)], bins=nbins, density=False)[0]
        x2s = np.zeros((nbins, nbins))
        for i in range(nbins):
            for j in range(nbins):
                x2s[i,j] = np.square(emp_freq[i,j] -  1//N*(np.sum(emp_freq[i,:])*np.sum(emp_freq[:,j])))
                x2s[i,j]/= 1//N*(np.sum(emp_freq[i,:])*np.sum(emp_freq[:,j]))
        coincnt += (x2<=np.sum(x2s[i,j]))
    pval = coincnt/nruns
    return x2, pval



######################################################################################################################
# Differential entropy
######################################################################################################################

def _numBins(nObs, corr=None):
    # source Machine learning for asset manager by Marcos M. Lopez de Prado 
    if corr is None:
        z = np.power(8+324*nObs+12*np.sqrt(36*nObs+729*nObs**2), 1//3)
        b = round(z//6 +2//(3*z)+1//3)
    else:
        b = round(np.power(2, -0.5)*np.sqrt(1+np.sqrt(1+24*nObs//(1-corr**2))))
    return int(b)

def differential_entropy(v1, v2, bins=None, norm=False):
    # source Machine learning for asset manager by Marcos M. Lopez de Prado 
    if isinstance(bins, int):
        bXY = bins    
    else:
        bXY = _numBins(v1.shape[0], corr=np.corrcoef(v1, v2)[0,1])
    cXY = np.histogram2d(v1, v2, bXY)[0]
    iXY = mutual_info_score(None, None, contingency=cXY)
    hX = sc.stats.entropy(np.histogram(v1, bins)[0])
    hY = sc.stats.entropy(np.histogram(v2, bins)[0])
    vXY = hX+hY-2*iXY
    if norm:
        hXY = hX+hY-iXY
        vXY /= hXY
    return vXY


######################################################################################################################
# Mutual Information
######################################################################################################################

def mutual_information(v1,v2, bins=None, norm=False):
    # source Machine learning for asset manager by Marcos M. Lopez de Prado 
    if isinstance(bins, int):
        bXY = bins    
    else:
        bXY = _numBins(v1.shape[0], corr=np.corrcoef(v1, v2)[0,1])
    cXY = np.histogram2d(v1, v2, bXY)[0]
    iXY = mutual_info_score(None, None, contingency=cXY)
    if norm:
        hX = sc.stats.entropy(np.histogram(x, bXY)[0])
        hY = sc.stats.entropy(np.histogram(y, bXY)[0])
        iXY /= min(hX, hY)
    return iXY


######################################################################################################################
# Heller–Heller–Gorfine
# https://academic.oup.com/biomet/article/100/2/503/202568
# https://master--mgc.netlify.app/_modules/mgcpy/independence_tests/hhg.html#HHG
######################################################################################################################

@jit(nopython=True, cache=True)
def _pearson_stat(distx, disty):  # pragma: no cover
    """Calculate the Pearson chi square stats"""

    n = distx.shape[0]
    S = np.zeros((n, n))

    # iterate over all samples in the distance matrix
    for i in range(n):
        for j in range(n):
            if i != j:
                a = distx[i, :] <= distx[i, j]
                b = disty[i, :] <= disty[i, j]

                t11 = np.sum(a * b) - 2
                t12 = np.sum(a * (1 - b))
                t21 = np.sum((1 - a) * b)
                t22 = np.sum((1 - a) * (1 - b))

                denom = (t11 + t12) * (t21 + t22) * (t11 + t21) * (t12 + t22)
                if denom > 0:
                    S[i, j] = ((n - 2) * (t12 * t21 - t11 * t22) ** 2) / denom
    return S


def HHG(X1,X2, metric="euclidean", njobs=4):
    distx = pairwise_distances(X1, metric=metric, n_jobs=4)
    disty = pairwise_distances(X2, metric=metric, n_jobs=4)

    S = _pearson_stat(distx, disty)
    mask = np.ones(S.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    stat = np.sum(S[mask])
    return stat 


######################################################################################################################
# Hilbert-Schmidt Independence Criterion
# https://arxiv.org/abs/1910.00270
# Does the joint-probability Pxy factorize as PxPy?
# https://github.com/amber0309/HSIC/blob/master/HSIC.py
# https://github.com/neurodata/hyppo/
######################################################################################################################


def rbf_dot(pattern1, pattern2, deg):
    size1 = pattern1.shape
    size2 = pattern2.shape

    G = np.sum(pattern1*pattern1, 1).reshape(size1[0],1)
    H = np.sum(pattern2*pattern2, 1).reshape(size2[0],1)

    Q = np.tile(G, (1, size2[0]))
    R = np.tile(H.T, (size1[0], 1))

    H = Q + R - 2* np.dot(pattern1, pattern2.T)

    H = np.exp(-H/2/(deg**2))

    return H

def HSIC2(X, Y, alph = 0.1):
    """
    X, Y are numpy vectors with row - sample, col - dim
    alph is the significance level
    auto choose median to be the kernel width
    """
    n = X.shape[0]

    # ----- width of X -----
    Xmed = X

    G = np.sum(Xmed*Xmed, 1).reshape(n,1)
    Q = np.tile(G, (1, n) )
    R = np.tile(G.T, (n, 1) )

    dists = Q + R - 2* np.dot(Xmed, Xmed.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n**2, 1)

    width_x = np.sqrt( 0.5 * np.median(dists[dists>0]) )
    # ----- -----

    # ----- width of X -----
    Ymed = Y

    G = np.sum(Ymed*Ymed, 1).reshape(n,1)
    Q = np.tile(G, (1, n) )
    R = np.tile(G.T, (n, 1) )

    dists = Q + R - 2* np.dot(Ymed, Ymed.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n**2, 1)

    width_y = np.sqrt( 0.5 * np.median(dists[dists>0]) )
    # ----- -----

    bone = np.ones((n, 1), dtype = float)
    H = np.identity(n) - np.ones((n,n), dtype = float) / n

    K = rbf_dot(X, X, width_x)
    L = rbf_dot(Y, Y, width_y)

    Kc = np.dot(np.dot(H, K), H)
    Lc = np.dot(np.dot(H, L), H)

    testStat = np.sum(Kc.T * Lc) / n

    varHSIC = (Kc * Lc / 6)**2

    varHSIC = ( np.sum(varHSIC) - np.trace(varHSIC) ) / n / (n-1)

    varHSIC = varHSIC * 72 * (n-4) * (n-5) / n / (n-1) / (n-2) / (n-3)

    K = K - np.diag(np.diag(K))
    L = L - np.diag(np.diag(L))

    muX = np.dot(np.dot(bone.T, K), bone) / n / (n-1)
    muY = np.dot(np.dot(bone.T, L), bone) / n / (n-1)

    mHSIC = (1 + muX * muY - muX - muY) / n

    al = mHSIC**2 / varHSIC
    bet = varHSIC*n / mHSIC

    thresh = gamma.ppf(1-alph, al, scale=bet)[0][0]

    return (testStat, thresh)


def compute_kern(x, y, metric="gaussian", workers=1, **kwargs):
    """
    Kernel similarity matrices for the inputs.
    Parameters
    ----------
    x,y : ndarray
        Input data matrices. ``x`` and ``y`` must have the same number of
        samples. That is, the shapes must be ``(n, p)`` and ``(n, q)`` where
        `n` is the number of samples and `p` and `q` are the number of
        dimensions. Alternatively, ``x`` and ``y`` can be kernel similarity matrices,
        where the shapes must both be ``(n, n)``.
    metric : str, callable, or None, default: "gaussian"
        A function that computes the kernel similarity among the samples within each
        data matrix.
        Valid strings for ``metric`` are, as defined in
        :func:`sklearn.metrics.pairwise.pairwise_kernels`,
            [``"additive_chi2"``, ``"chi2"``, ``"linear"``, ``"poly"``,
            ``"polynomial"``, ``"rbf"``,
            ``"laplacian"``, ``"sigmoid"``, ``"cosine"``]
        Note ``"rbf"`` and ``"gaussian"`` are the same metric.
        Set to ``None`` or ``"precomputed"`` if ``x`` and ``y`` are already similarity
        matrices. To call a custom function, either create the similarity matrix
        before-hand or create a function of the form :func:`metric(x, **kwargs)`
        where ``x`` is the data matrix for which pairwise kernel similarity matrices are
        calculated and kwargs are extra arguements to send to your custom function.
    workers : int, default: 1
        The number of cores to parallelize the p-value computation over.
        Supply ``-1`` to use all cores available to the Process.
    **kwargs
        Arbitrary keyword arguments provided to
        :func:`sklearn.metrics.pairwise.pairwise_kernels`
        or a custom kernel function.
    Returns
    -------
    simx, simy : ndarray
        Similarity matrices based on the metric provided by the user.
    """
    if not metric:
        metric = "precomputed"
    if metric in ["gaussian", "rbf"]:
        if "gamma" not in kwargs:
            l2 = pairwise_distances(x, metric="l2", n_jobs=workers)
            n = l2.shape[0]
            # compute median of off diagonal elements
            med = np.median(
                np.lib.stride_tricks.as_strided(
                    l2, (n - 1, n + 1), (l2.itemsize * (n + 1), l2.itemsize)
                )[:, 1:]
            )
            # prevents division by zero when used on label vectors
            med = med if med else 1
            kwargs["gamma"] = 1.0 / (2 * (med ** 2))
        metric = "rbf"
    simx = pairwise_kernels(x, metric=metric, n_jobs=workers, **kwargs)
    simy = pairwise_kernels(y, metric=metric, n_jobs=workers, **kwargs)
    return simx, simy

@jit(nopython=True, cache=True)
def _center_distmat(distx, bias):  # pragma: no cover
    """Centers the distance matrices"""
    n = distx.shape[0]
    if bias:
        # use sum instead of mean because of numba restrictions
        exp_distx = (
            np.repeat(distx.sum(axis=0) / n, n).reshape(-1, n).T
            + np.repeat(distx.sum(axis=1) / n, n).reshape(-1, n)
            - (distx.sum() / (n * n))
        )
    else:
        exp_distx = (
            np.repeat((distx.sum(axis=0) / (n - 2)), n).reshape(-1, n).T
            + np.repeat((distx.sum(axis=1) / (n - 2)), n).reshape(-1, n)
            - distx.sum() / ((n - 1) * (n - 2))
        )
    cent_distx = distx - exp_distx
    if not bias:
        np.fill_diagonal(cent_distx, 0)
    return cent_distx

@jit(nopython=True, cache=True)
def _cpu_cumsum(data):  # pragma: no cover
    """Create cumulative sum since numba doesn't sum over axes."""
    cumsum = data.copy()
    for i in range(1, data.shape[0]):
        cumsum[i, :] = data[i, :] + cumsum[i - 1, :]
    return cumsum


@jit(nopython=True, cache=True)
def _fast_1d_dcov(x, y, bias=False):  # pragma: no cover
    """
    Calculate the Dcorr test statistic. Note that though Dcov is calculated
    and stored in covar, but not called due to a slower implementation.
    See: https://www.sciencedirect.com/science/article/abs/pii/S0167947319300313
    """
    n = x.shape[0]

    # sort inputs
    x_orig = x.ravel()
    x = np.sort(x_orig)
    y = y[np.argsort(x_orig)]
    x = x.reshape(-1, 1)  # for numba

    # cumulative sum
    si = _cpu_cumsum(x)
    ax = (np.arange(-(n - 2), n + 1, 2) * x.ravel()).reshape(-1, 1) + (si[-1] - 2 * si)

    v = np.hstack((x, y, x * y))
    nw = v.shape[1]

    idx = np.vstack((np.arange(n), np.zeros(n))).astype(np.int64).T
    iv1 = np.zeros((n, 1))
    iv2 = np.zeros((n, 1))
    iv3 = np.zeros((n, 1))
    iv4 = np.zeros((n, 1))

    i = 1
    r = 0
    s = 1
    while i < n:
        gap = 2 * i
        k = 0
        idx_r = idx[:, r]
        csumv = np.vstack((np.zeros((1, nw)), _cpu_cumsum(v[idx_r, :])))

        for j in range(1, n + 1, gap):
            st1 = j - 1
            e1 = min(st1 + i - 1, n - 1)
            st2 = j + i - 1
            e2 = min(st2 + i - 1, n - 1)

            while (st1 <= e1) and (st2 <= e2):
                idx1 = idx_r[st1]
                idx2 = idx_r[st2]

                if y[idx1] >= y[idx2]:
                    idx[k, s] = idx1
                    st1 += 1
                else:
                    idx[k, s] = idx2
                    st2 += 1
                    iv1[idx2] += e1 - st1 + 1
                    iv2[idx2] += csumv[e1 + 1, 0] - csumv[st1, 0]
                    iv3[idx2] += csumv[e1 + 1, 1] - csumv[st1, 1]
                    iv4[idx2] += csumv[e1 + 1, 2] - csumv[st1, 2]
                k += 1

            if st1 <= e1:
                kf = k + e1 - st1 + 1
                idx[k:kf, s] = idx_r[st1 : e1 + 1]
                k = kf
            elif st2 <= e2:
                kf = k + e2 - st2 + 1
                idx[k:kf, s] = idx_r[st2 : e2 + 1]
                k = kf

        i = gap
        r = 1 - r
        s = 1 - s

    covterm = np.sum(n * (x - np.mean(x)).T @ (y - np.mean(y)))
    c1 = np.sum(iv1.T @ v[:, 2].copy())
    c2 = np.sum(iv4)
    c3 = np.sum(iv2.T @ y)
    c4 = np.sum(iv3.T @ x)
    d = 4 * ((c1 + c2) - (c3 + c4)) - 2 * covterm

    y_sorted = y[idx[n::-1, r], :]
    si = _cpu_cumsum(y_sorted)
    by = np.zeros((n, 1))
    by[idx[::-1, r]] = (np.arange(-(n - 2), n + 1, 2) * y_sorted.ravel()).reshape(
        -1, 1
    ) + (si[-1] - 2 * si)

    if bias:
        denom = [n ** 2, n ** 3, n ** 4]
    else:
        denom = [n * (n - 3), n * (n - 3) * (n - 2), n * (n - 3) * (n - 2) * (n - 1)]

    stat = np.sum(
        (d / denom[0])
        + (np.sum(ax) * np.sum(by) / denom[2])
        - (2 * (ax.T @ by) / denom[1])
    )

    return stat

@jit(nopython=True, cache=True)
def _dcov(distx, disty, bias=False, only_dcov=True):  # pragma: no cover
    """Calculate the Dcov test statistic"""
    if only_dcov:
        # center distance matrices
        distx = _center_distmat(distx, bias)
        disty = _center_distmat(disty, bias)

    stat = np.sum(distx * disty)

    if only_dcov:
        N = distx.shape[0]
        if bias:
            stat = 1 / (N ** 2) * stat
        else:
            stat = 1 / (N * (N - 3)) * stat

    return stat

@jit(nopython=True, cache=True)
def _dcorr(distx, disty, bias=False, is_fast=False):  # pragma: no cover
    """
    Calculate the Dcorr test statistic.
    """
    if is_fast:
        # calculate covariances and variances
        covar = _fast_1d_dcov(distx, disty, bias=bias)
        varx = _fast_1d_dcov(distx, distx, bias=bias)
        vary = _fast_1d_dcov(disty, disty, bias=bias)
    else:
        # center distance matrices
        distx = _center_distmat(distx, bias)
        disty = _center_distmat(disty, bias)

        # calculate covariances and variances
        covar = _dcov(distx, disty, bias=bias, only_dcov=False)
        varx = _dcov(distx, distx, bias=bias, only_dcov=False)
        vary = _dcov(disty, disty, bias=bias, only_dcov=False)

    # stat is 0 with negative variances (would make denominator undefined)
    if varx <= 0 or vary <= 0:
        stat = 0

    # calculate generalized test statistic
    else:
        stat = covar / np.real(np.sqrt(varx * vary))

    return stat

def HSIC(X, Y):
    kernx, kerny = compute_kern(X, Y, metric="gaussian")
    distx = 1 - kernx / np.max(kernx)
    disty = 1 - kerny / np.max(kerny)
    return _dcorr(distx, disty, bias=True, is_fast=True)

#######################################################################################################################
# Hellinger distance; 
# H =sqrt(sum(sqrt(p(x)*q(x))-1)
# H = 1/sqrt(2)*||sqrt(p)-sqrt(q)||
# H = sqrt(1 - _Bhattacharyya(P,Q))
#######################################################################################################################

@jit
def _Hellinger(v1, v2, nbins=20, how='bhatta'):
    if how=='bhatta':
        return np.sqrt(1-_Bhattacharyya(v1, v2, nbins=nbins))
    else:
        #totnum = v1.shape[0]
        _, edges = np.histogram(np.hstack((v1,v2)).flatten(), bins=nbins)
        d1 = np.histogram(v1, bins=edges, density=True)[0]
        d2 = np.histogram(v2, bins=edges, density=True)[0]
        #v1probs = np.bincount(d1)//totnum
        #v2probs = np.bincount(d2)//totnum
        return np.sqrt(1-np.sum(d1*d2))


#######################################################################################################################
# Bhattacharyya distance;  Db = -ln(sum(sqrt(p(x)*q(x))))
#######################################################################################################################

def _Bhattacharyya(v1,v2, nbins=20):
    # get ranges r0 to rM
    # get counts per distributions for each range, c_v1(ri), c_v2(ri) for i = 0..M
    _, edges = np.histogram(np.hstack((v1,v2)).flatten(), bins=nbins)
    d1 = np.histogram(v1, bins=edges, density=True)[0]
    d2 = np.histogram(v2, bins=edges, density=True)[0]
    #brgns = list(zip(edges[:-1], edges[1:]))
    #v1bins, v2bins = _get_bin_counts(v1,v2, brgns)
    #keep = np.argwhere(np.abs(v1bins+v2bins)>0)
    return -np.log(np.sum(np.sqrt(d1*d2)))

def Bhattacharyya_distance(X, 
                           sparse=False,
                           nbins=10, 
                           sparse_threshold_ht=0.9, 
                           sparse_threshold_lt=0.1):
    '''
        Assumes column * column
    '''
    assert len(X.shape)==2, ' X should be a 2D-dimensional array'
    ncols = X.shape[1]

    if sparse==False:
        dists = np.zeros((ncols, ncols))
        for c1 in range(ncols):
            v1 = X[:, c1]
            for c2 in range(c1+1, ncols):
                v2 = X[:, c2]
                dists[c1,c2] = _Bhattacharyya(v1, v2, nbins=nbins)
        dists = np.triu(dists) + np.tril(dists.T,1)
    else:
        data = []
        rows = []
        cols = []
        for c1 in range(ncols):
            v1 = X[:, c1]
            for c2 in range(c1+1, ncols):
                v2 = X[:, c2]
                distance=_Bhattacharyya(v1, v2, nbins=nbins)
                if (distance>sparse_threshold_ht) | (distance<sparse_threshold_lt):
                    data.append(distance)
                    rows.append(c1)
                    cols.append(c2) 
        dists = sc.sparse.csc_matrix((data, (rows, cols)), shape=(ncols, ncols), dtype=np.float32)
    return dists



#######################################################################################################################
# Total variation distance
# largest possible difference between the probabilities that the two pdf can assign to
# the same event
# NOT CORRECT AT THE MOMENT
#######################################################################################################################

def _TVD(v1,v2, nbins=20):
    # get ranges r0 to rM
    # get counts per distributions for each range, c_v1(ri), c_v2(ri) for i = 0..M
    _, edges = np.histogram(np.hstack((v1,v2)).flatten(), bins=nbins)
    brgns = list(zip(edges[:-1], edges[1:]))
    v1bins, v2bins = _get_bin_counts(v1,v2, brgns)
    return np.max(np.abs(v1bins-v2bins))

#######################################################################################################################
# Chi2-distance
# NOT CORRECT AT THE MOMENT
#######################################################################################################################

def _get_bin_counts(v1,v2,brgns):
    v1l, v2l = np.zeros(len(brgns)), np.zeros(len(brgns), dtype=np.int32)
    c1, c2 = len(v1), len(v2)
    for i,r in enumerate(brgns):
        v1l[i] = sum((v1>=r[0]) & (v1<=r[1]))
        v2l[i] = sum((v2>=r[0]) & (v2<=r[1]))
    return v1l/c1, v2l/c2

def _Chi2Distance(v1,v2, nbins=10, eps=1e-3):
    # get ranges r0 to rM
    # get counts per distributions for each range, c_v1(ri), c_v2(ri) for i = 0..M
    _, edges = np.histogram(np.hstack((v1,v2)).flatten(), bins=nbins)
    brgns = list(zip(edges[:-1], edges[1:]))
    v1bins, v2bins = _get_bin_counts(v1,v2, brgns)
    keep = np.argwhere(np.abs(v1bins+v2bins)>0)
    ch2d = np.sum((np.square(v1bins[keep]-v2bins[keep]))/(v1bins[keep]+v2bins[keep]))
    return ch2d

def Chi2Distance(X, sparse=False, nbins=10, sparse_threshold_ht=0.9, sparse_threshold_lt=0.1):
    '''
        Assumes column * column
    '''
    assert len(X.shape)==2, ' X should be a 2D-dimensional array'
    ncols = X.shape[1]

    print(f'X-shape:{X.shape}')

    if sparse==False:
        dists = np.zeros((ncols, ncols))
        for c1 in range(ncols):
            v1 = X[:, c1]
            for c2 in range(c1+1, ncols):
                v2 = X[:, c2]
                dists[c1,c2] = _Chi2Distance(v1, v2, nbins=nbins)                
        dists = np.triu(dists) + np.tril(dists.T,1)
    else:
        data = []
        rows = []
        cols = []
        for c1 in range(ncols):
            v1 = X[:, c1]
            for c2 in range(c1+1, ncols):
                v2 = X[:, c2]
                distance=_Chi2Distance(v1, v2, nbins=nbins)
                if (distance>sparse_threshold_ht) | (distance<sparse_threshold_lt):
                    data.append(distance)
                    rows.append(c1)
                    cols.append(c2) 
        dists = sc.sparse.csc_matrix((data, (rows, cols)), shape=(ncols, ncols), dtype=np.float32)
    return dists

#######################################################################################################################
# Mahalanobis distance;  cov, invcov = _cov(x, inverse=True)  -> _Mahalanobis(v1, v2, inv_cov)
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
def _Mahalanobis(v1, v2, inv_cov=None):
    return sc.spatial.distance.mahalanobis(v1, v2, inv_cov)

def Mahalanobis(X, v1=None, v2=None, featurewise=True):
    '''
        X: numpy array, rows contains samples, columns contains features
        v1: column index 
        v2: column index
    '''
    if featurewise:
        X = np.transpose(X) 
    _, icov = _cov(X, inverse=True)
    if v1 is None:
        # all v. all
        return sc.spatial.pdist(X, metric='mahalanobis', VI=icov)
    elif v2 is None:
        # v1 v. the X features
        return sc.spatial.cdist(X,X[v2,:], metric='mahalanobis', VI=icov)
    else:
        # v1 v. v2
        return _Mahalanobis(X[v1,:], X[v2,:], icov)


#######################################################################################################################
# Maximal Correlation Analysis (MAC)
# http://data.bit.uni-bonn.de/publications/ICML2014.pdf
# Use MAC to find non-linearly related features, similar to distcorr and mic.
#######################################################################################################################



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
# Likelihood ratio test
# goodness of fit using the frequenties
#######################################################################################################################

def powerdiv_scores(X,y, bins=10, pdivtype='llr'):
    if pdivtype=='llr':
        _lambda = 0.
    elif pdivtype=='mllr':
        _lambda = -1.
    elif pdivtype=='neyman':
        _lambda = -2.
    elif pdivtype=='pearson':
        _lambda = 1.
    elif pdivtype=='cressie-read':
        _lambda = 2//3
    elif pvdivtype=='freeman-tukey':
        _lambda = -1//2


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


    
######################################################################################################################
# Statistical pairwise difference
# Kullback-Leibler, Cross-entropy, Jensen-Shannon, Pearson, Spearman, Kendal-tau
# paired t-test, Wilcoxon signed-rank
######################################################################################################################
# peardist,pearpval = pearsonr(v1,v2)
# speardist,spearpval = spearmanr(v1,v2)
# kendalldist,kendallpval = kendalltau(v1,v2)
# wtaudist,wtaupval = weightedtau(v1,v2)
# wilcoxstat, wilcoxpval = sc.stats.wilcoxon(v1,v2)
# tteststat, ttestpval = sc.stats.ttest_rel(v1,v2)

######################################################################################################################
# Statistical unpaired distance
# mannwhitneyu
# student t-test
# Welch t-test
# kolmogorov-smirnov
# epps
# Cramer-VonMises
# Wasserstein n^th
######################################################################################################################


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


#######################################################################################################################
# RV-coefficient
# trace(sum(xy)*sum(yx))/trace(sum2xx)/trace(sum2yy)
#######################################################################################################################
@jit
def RV_coefficient(X1,X2):
    covar = np.dot(X1.T, X2)
    varX = np.dot(X1.T, X1)
    varY = np.dot(X2.T, X2)
    covar2 = np.trace(np.dot(covar, covar.T))
    varX2 = np.trace(np.dot(varX, varX))
    varY2 = np.trace(np.dot(varY, varY))
    rv_coeff = np.divide(covar2, np.sqrt(varX2*varY2))   
    return rv_coeff

#######################################################################################################################
# Congruence coefficient
#######################################################################################################################

@jit
def congruence_coefficient(v1,v2):
    return np.sum(v1*v2)/np.sqrt(np.sum(np.square(v1))*np.sum(np.square(v2)))