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


"""
     Information change: 
        * Kullback-Leibler divergence (have to make same number of bins)
        * cross-entropy
        * Shannon entropy change
"""
@jit
def _information_change(v1, v2, ent_type = 'kl', bin_type='fixed', num_bins=25):
    '''
    v1: vector one
    v2: vector two
    ent_type : kl, shannon, cross
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

    

def _cdf(x,bin_size=5):
    x = np.sort(x)
    c = len(x)
    res, _res= np.empty((0, 2)), np.empty((0,2))
    for i in range(bin_size, c):
        if i%bin_size==0:
            _res = np.array([i/c, np.median(x[(i-bin_size):i])])
            res = np.append(res, [_res], axis=0)    
    return res
# np.asarray([[i/c, np.median(x[(i-bin_size):i])] for i in range(bin_size, c) if i%bin_size==0])

@jit
def _cdfcoeff(x, bin_size=5):
    # better to take out the creation of the cdf and split the function (_cdf, _cdfcoeff)
    lt = _cdf(x, bin_size=bin_size)
    xmm = x.max()-x.min()
    diff1 = np.diff(lt[:,1])*lt[:-1, 0]
    diff1[0:1] = 0  
    diff2 = np.diff(diff1)
    diff2[0:1] = 0 
    diff2 = diff2*lt[:-2, 0]

    # modus is peak in f'' followed by valley  in f''
    xmmp = xmm**4
    tv, td = (diff2[0:-1]-diff2[1:]), (np.sign(diff2[0:-1])-np.sign(diff2[1:]))
    xrelcount, xbumpmean, xdiffmean, xdiffvar = (td>0).sum()/(lt.shape[0]-1), tv[tv>0].mean()/xmmp, diff1.mean()/xmmp, diff1.std()/xmmp
    
    return xrelcount, xbumpmean, xdiffmean, xdiffvar

def _interp(xinter, yinter, xextra):
    return minterp(xinter, yinter, axis=0, extrapolate=True)(xextra)

def _cdf_distanceB(x1, x2, bin_size=5, minkowsi=1):
    '''
     takes the Minkowski d492istance between the ecdf's and the Russell-Rao distance of the bump indicators
    ''' 
    lt1 = _cdf(x1, bin_size=bin_size)
    lt2 = _cdf(x2, bin_size=bin_size)

    l1 = _interp(xinter=lt1[:,0], yinter=lt1[:,1], xextra=lt1[:,0])
    l2 = _interp(xinter=lt2[:,0], yinter=lt2[:,1], xextra=lt1[:,0])

    l1diff = np.diff(l1)
    l2diff = np.diff(l2)

    l1diff2 = np.diff(l1diff)
    l2diff2 = np.diff(l2diff)

    l1diff2sign = np.sign(l1diff2)
    l2diff2sign = np.sign(l2diff2)

    l1bump = l1diff2sign[0:-1]-l1diff2sign[1:]
    l2bump = l2diff2sign[0:-1]-l2diff2sign[1:] 

    d1 = sc.spatial.distance.minkowski(l1, l2, p=minkowsi)*sc.spatial.distance.russellrao(l1bump, l2bump)
    d2 = sc.spatial.distance.minkowski(l1diff, l2diff, p=minkowsi)*sc.spatial.distance.russellrao(l1bump, l2bump)
    d3 = sc.spatial.distance.russellrao(l1bump, l2bump)   
    return d1,d2,d3

def _cdf_distanceG(x1, x2, bin_size=15):
    # also see PhD-thesis Gabriel Martos Venturini, Statistical distance and probability metrics for multivariate data..etc., June 2015 Uni. Carlos III de Madrid
    # https://core.ac.uk/download/pdf/30276753.pdf
    '''
     Basically the l1-difference between the cdf's
    '''
    lt1 = _cdf(x1, bin_size=bin_size)
    lt2 = _cdf(x2, bin_size=bin_size)

    l1 = _interp(xinter=lt1[:,0], yinter=lt1[:,1], xextra=lt1[:,0])
    l2 = _interp(xinter=lt2[:,0], yinter=lt2[:,1], xextra=lt1[:,0])    
    
    c = np.max([l1.max(), l2.max()]) - np.min([l1.min(), l2.min()])
    
    diffSumabsAgg= np.sum(lt1[:,0]*(np.abs(l2-l1)))/c # qEMD
    diffSumSqAgg = np.sqrt(np.sum(lt1[:,0]*np.power((l2-l1), 2)))/c # qCvM
    diffSumAgg = np.sum(lt1[:,0]*(l2-l1))/c
    return diffSumAgg, diffSumSqAgg, diffSumabsAgg

def _scaled_stat_distance(x1, x2, test='KS'):
    # scaled with difference between median's
    mdiff = np.median(x2)-np.median(x1)
    return mdiff*stats.ks_2samp(x1, x2)[0]
