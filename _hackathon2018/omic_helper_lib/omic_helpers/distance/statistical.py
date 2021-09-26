import numpy as np
import itertools
from numba import jit, njit
import torch
import scipy as sc

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
def _get_pairs(v1,v2):
    K = int(len(v1)**2/2-len(v1))
    pairs = np.zeros((K,4))
    for i in range(len(V1)):
        for j in range(i, range(len(V1))):
            pairs[k, :] = np.hstack([v1[i,:], v2[j,:]])
    return pairs

def _get_concordant_pairs(v1, v2):
    _pairs = _get_pairs(v1,v2)
    return sum((np.sign(_pairs[:,0]-_pairs[:,2]) == 
                    np.sign(_pairs[:,1]-_pairs[:,3])).astype(np.int32))

def _get_discordant_pairs(v1, v2):
    _pairs = _get_pairs(v1,v2)
    return sum((np.sign(_pairs[:,0]-_pairs[:,2]) != 
                    np.sign(_pairs[:,1]-_pairs[:,3])).astype(np.int32))
    
def _get_tied_pairs(v1, v2):
    return pairs    

def goodman_kruskal_gamma(v1,v2):
    assert isinstance(v1, np.array), 'v1 is not a numpy array'
    assert isinstance(v2, np.array), 'v2 is not a numpy array'
    assert v1.shape == v2.shape, 'Shapes of v1, v2 should be the same'
    assert len(v1.shape)>=1, 'v1 is not an Nd array'
    assert len(v2.shape)>=1, 'v2 is not an Nd array'
    
    # G = (Nc - Nd)/(Nc+Nd)
    # Nc = # of concordant pairs
    # Nd = # of discordant pairs
    # pairs are concordant, discordant or tied
    K = int(len(v1)**2/2-len(v1))
    Nc = _get_concordant_pairs(v1,v2)
    Nd = K - Nc    
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

def differential_entropy(v1,v2, bins=None, norm=False):
    # source Machine learning for asset manager by Marcos M. Lopez de Prado 
    if isinstance(bins, int):
        bXY = bins    
    else:
        bXY = _numBins(v1.shape[0], corr=np.corrcoef(v1, v2)[0,1])
    cXY = np.histogram2d(x,y, bXY)[0]
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
######################################################################################################################




######################################################################################################################
# Hilbert-Schmidt Independence Criterion
# https://arxiv.org/abs/1910.00270
######################################################################################################################


def pairwise_distances(x):
    #x should be two dimensional
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ /sigma)

def HilbertSchmidt(x, y, s_x=1, s_y=1):
    m,_ = x.shape #batch size
    K = GaussianKernelMatrix(x,s_x)
    L = GaussianKernelMatrix(y,s_y)
    H = torch.eye(m) - 1.0/m * torch.ones((m,m))
    H = H.double()
    HSIC = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
    return HSIC

#######################################################################################################################
# Hellinger distance; 
# H =sqrt(sum(sqrt(p(x)*q(x))-1)
# H = 1/sqrt(2)*||sqrt(p)-sqrt(q)||
# H = sqrt(1 - _Bhattacharyya(P,Q))
#######################################################################################################################

@jit
def _Hellinger(v1, v2, num_bins=20):
    totnum = v1.shape[0]
    _, edges = np.histogram(np.hstack((v1,v2)).flatten(), bins=num_bins)
    d1 = np.digitize(v1, edges)
    d2 = np.digitize(v2, edges)
    v1probs = np.bincount(d1)//totnum
    v2probs = np.bincount(d2)//totnum
    
    


#######################################################################################################################
# Bhattacharyya distance;  Db = -ln(sum(sqrt(p(x)*q(x))))
#######################################################################################################################

def _Bhattacharyya(v1,v2, nbins=10):
    # get ranges r0 to rM
    # get counts per distributions for each range, c_v1(ri), c_v2(ri) for i = 0..M
    _, edges = np.histogram(np.hstack((v1,v2)).flatten(), bins=nbins)
    brgns = list(zip(edges[:-1], edges[1:]))
    v1bins, v2bins = _get_bin_counts(v1,v2, brgns)
    keep = np.argwhere(np.abs(v1bins+v2bins)>0)
    return -np.log(np.sum(np.sqrt(v1bins[keep]*v2bins[keep])))

def Bhattacharyya_distance(X, sparse=False, nbins=10, sparse_threshold_ht=0.9, sparse_threshold_lt=0.1):
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
# Chi2-distance
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

@jit
def _Mahalanobis(v1, v2, inv_cov):
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
