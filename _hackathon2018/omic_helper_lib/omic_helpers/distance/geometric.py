'''
X,y --> dist.fit_transform(X,y) 

'''


#######################################################################################################################
# Distance Correlation: https://stats.stackexchange.com/questions/183572/understanding-distance-correlation-computations,
# https://gist.github.com/satra/aa3d19a12b74e9ab7941
# https://github.com/vnmabus/dcor
#
# Purpose: to compare different features for non-linear similarity
#######################################################################################################################

from scipy.spatial.distance import pdist, squareform
from numba import jit, float32
from joblib import Parallel, delayed

# dcor_cor = dcor.distance_correlation_sqrt(v1, v2, exponent=0.5,  method='AVL')
# dcor_pval = dcor.distance_correlation_t_test(v1,v2)
# dcoru_cor = dcor.u_distance_correlation_sqrt(v1, v2, exponent=0.5,  method='AVL')

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

    TODO: add bootstrap option for p-value estimation : https://gist.github.com/raphaelvallat/386da9eff858b8bd2e647bc1be4c7566
    TODO: if not performant; use https://dcor.readthedocs.io/en/latest/performance.html#paralllel-computation-of-distance-covariance
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


<<<<<<< HEAD
#######################################################################################################################
# Chord distance 
#######################################################################################################################



#######################################################################################################################
# Mean character distance
#######################################################################################################################



#######################################################################################################################
# Index of association
#######################################################################################################################



#######################################################################################################################
# Coefficient of divergence
#######################################################################################################################


#######################################################################################################################
# Czekanowski coefficient
#######################################################################################################################
=======
################################################################################
# Chord distance
################################################################################




################################################################################
# Mean character distance
################################################################################



################################################################################
# Index of association
################################################################################



################################################################################
# Coefficient of divergence
################################################################################



################################################################################
# Czekanowski coefficient
################################################################################
>>>>>>> master
