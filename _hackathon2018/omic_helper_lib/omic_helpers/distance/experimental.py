import numpy as np

######################################################################################################################
# Mean Absolute Piecewise Similarity (novel)
# non-overlapping 2D patches with some similarity metric
######################################################################################################################
def MAPS(v1,v2, scorer=pearsonr, min_samples=100, min_percentage=0.25, n_iters=100, convolve=False):
    '''
    Local
         - (Normalise)
         - Make 2D patches: max(10%_samples, min_samples),
         - Recenter per patch
         - per patch determine correlation score
         - return mean statistic and mean nlog of p-value
         - with/without convolution
    '''

    assert np.max([min_percentage*len(v1), min_samples]) < len(v1), f'Lower min_samples to {min_percentage*len(v1)}'
    # make patches
    sx, sy = np.quantile(v1, q=np.arange(0,1,min_percentage)), np.quantile(v2, q=np.arange(0,1,min_percentage))
    sx = np.insert(sx,0,np.min(v1)); sx = np.insert(sx,-1,np.max(v1))
    sy = np.insert(sy,0,np.min(v2)); sy = np.insert(sy,-1,np.max(v2))
    patch_list = []
    for i in range(len(sx)-1):
        v1indcs = np.argwhere((v1>sx[i]) & (v1<sx[i+1]))[0]
        for j in range(len(sy)-1):
            v2indcs = np.argwhere((v2>sx[i]) & (v2<sx[i+1]))[0]
            patch_list.append((v1[v1indcs], v2[v2indcs]))
    # cycle through patches

    # get stats

    return maps_score, p_value


######################################################################################################################
# Continuous Rule Combination
# The goal is to identify non-bi-normal distributions
# - Sum of angles = 0
# - rotational invariance of variance
# - unimodality
# first, perform quantile normalisation
######################################################################################################################

def CoRuCo(v1, v2):
    

    return True


######################################################################################################################
# Split Correlation Estimate (SCorE). (novel)
# (supervised) Score is 1 - (number of splits needed for perfect separation) divided by minority class count
# (unsupervised) Score is 1 -  (number of splits needed for perfection separation of binarised v2 using v1) 
# divided by minority class count
######################################################################################################################

def SCorE(v1, v2, bins_fit=(2,100), bins=100):
    qs = np.arange(0,1,bins)
    qbins2 = np.quantile(v2, qs)
    bins2 = np.digitize(v2, qbins2)

    for b in range(bins_fit):
        qs = np.arange(0,1,b)
        qbins1 = np.quantile(v2,qs)
        bins1 = np.digitize(v1,qbins1)
        
    return True

######################################################################################################################
# "Power Predictive Score"
# Basically; how well does A predict B using a non-linear predictor, based on cross-validated scores
######################################################################################################################
'''
 f(A) -> B, N-fold CV, average F1/MCC on OOF
'''
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.svm import SVR, SVC
from sklearn.linear_model import HuberRegressor, LogisticRegression, Lasso
from sklearn.metrics import f1_score, matthews_corrcoef
def PPS(x,y, num_folds: int=10, num_iter: int=10, clf_type: str='regressor', robust: bool=True):
    ''' Predictive power score, assumes RMSE as metric, assumes regressor, or binomial
    '''
    Kfolder = RepeatedKFold(n_splits=num_folds, n_repeats=num_iter)
    MCCs = []
    F1s = []
    CORRS = []
    if clf_type == 'regressor':
        if robust:
            mod = HuberRegressor()
        else:
            mod = SVR()
    else:
        if robust:
            mod = LogisticRegression(penalty='l1', solver='liblinear')
        else:
            mod = SVC()

    if len(x.shape)==1:
        x = x.reshape((-1,1))

    if clf_type == 'regressor':
        for train,test in Kfolder.split(x,y):
            mod.fit(x[train], y[train])
            y_pred = mod.predict(x[test])
            CORRS.append(spearmanr(y[test], y_pred)[0])
        return np.nanmean(CORRS), None
    else:
        for train,test in Kfolder.split(x,y):
            mod.fit(x[train], y[train])
            y_pred = mod.predict(x[test])
            MCCs.append(f1_score(y[test], y_pred))
            F1s.append(matthews_corrcoef(y[test], y_pred))
        return np.nanmean(MCCs), np.nanmean(F1s)