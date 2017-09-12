from sklearn import preprocessing, svm, tree, ensemble, naive_bayes, neural_network, model_selection, metrics
from sklearn import discriminant_analysis, decomposition, cross_decomposition
import numpy as np
import pandas as pd
from collections import Counter 
from math import*
from scipy.spatial.distance import minkowski
from scipy.spatial.distance import cdist
from decimal import Decimal
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from time import time
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt
from itertools import cycle


def _group_patients(df, method = 'first'): # method = ['first', 'mean', 'median', 'min', 'max']
    # 1. clean probesets 
    # 2. reduce multiple probesets per gene to one  
    df_1 = df[df.columns[:21]].groupby("labnr patient").apply(lambda g: g.iloc[0])
    col_list = df.columns[21:].values.tolist()
    col_list.append("labnr patient")
    if(method == 'first'):
        df_2 = df[col_list].groupby("labnr patient").apply(lambda g: g.iloc[0])
    elif(method == 'mean'):
        df_2 = df[col_list].groupby("labnr patient").mean()
    elif(method == 'median'):
        df_2 = df[col_list].groupby("labnr patient").median()
    elif(method == 'min'):
        df_2 = df[col_list].groupby("labnr patient").min()
    elif(method == 'max'):
        df_2 = df[col_list].groupby("labnr patient").max()  
    dfinal = df_1.merge(df_2, left_index = True, right_index = True)

    return dfinal

def _cohort_correction(df):
    ## TO FINISH
    # correct for bias introduced by measurements
    # check for means in measurement groups, for similar patients (use groups from affinity propagation?)
    return True

def get_patient_similarity(patient_matrix, sim_type = 'cosine', minkowski_dim = None, normalised = True, inflation = 1):
    ''' Function to get similarity measures between patients  
        Variables:
            patient_matrix : dataframe with patient 1..N as columns and genome expressions 1..M as rows.
            sim_type : type of similarity measure
                values : 'cosine', 'manhattan', 'euclidian', 'minkowski', 'kendall', 'spearman', 'pearson',
                ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘dice’,‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, 
                ‘matching’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’
            minkowski_dim : dimensionality of minkowski space
                values : 3,4...inf (1= is manhattan, 2=euclidian)
            normalised : boolean
        output : 
            similarity matrix (DataFrame)       
    '''

    if (sim_type in ['cosine', 'manhattan', 'euclidian', 'minkowski', 'braycurtis', 'canberra', 'chebyshev', 'dice','hamming', 'jaccard', 'kulsinski', 'mahalanobis', 
                'matching', 'rogerstanimoto', 'russellrao', 'seuclidean', '‘sokalmichener', 'sokalsneath']):
        A = numpy.array(patient_matrix)
    else:
        A = None
    VarList = patient_matrix.T.keys()
    if(sim_type == 'cosine'):        
        similarities = cdist(A, A, metric = 'cosine')
        patient_similarity = pandas.DataFrame(similarities, index = VarList, columns = VarList)
    elif(sim_type == 'manhattan'):
        similarities = cdist(A, A, metric = 'cityblock')
        patient_similarity = pandas.DataFrame(similarities, index = VarList, columns = VarList)        
    elif(sim_type == 'euclidian'):
        similarities = cdist(A, A, metric = 'euclidean')
        patient_similarity = pandas.DataFrame(similarities, index = VarList, columns = VarList)
    elif(sim_type == 'minkowski'):
        if (minkowski_dim == None):
            print("No Minkowski dimension given! Assuming minkowski dim is 3")
            minkowski_dim = 3
        similarities = cdist(A, A, metric = 'minkowski', p = minkowski_dim)
        patient_similarity = pandas.DataFrame(similarities, index = VarList, columns = VarList)       
    elif(sim_type in ['kendall', 'spearman', 'pearson']):
        patient_similarity = 1-patient_matrix.T.astype('float64').corr(method = sim_type)    
    elif(sim_type in ['braycurtis', 'canberra', 'chebyshev', 'dice','hamming', 'jaccard', 'kulsinski', 'mahalanobis', 
                'matching', 'rogerstanimoto', 'russellrao', 'seuclidean', '‘sokalmichener', 'sokalsneath']):
        similarities = cdist(A, A, metric = sim_type)
        patient_similarity = pandas.DataFrame(similarities, index = VarList, columns = VarList)       
    ###
    if inflation > 1:
        patient_similarity = patient_similarity**inflation
    ###                                                               
    if normalised == True: # ! IMPROVE, not memory efficient
        patient_similarity = (patient_similarity - min(patient_similarity.min()))/(max(patient_similarity.max())-min(patient_similarity.min()))
    
    patient_similarity = 1 - patient_similarity
    ###
    return patient_similarity

def get_genome_similarity(df, reduction = 'filtered', max_dim = 10000):
    ## TO FINISH, low-dim, high number of vectors
    ## need to create sparse representation otherwise we end up with a 55.000 x 55.000 matrix
    df_reduced = _get_reduced(df, reduce_type= reduction)


    return True

def get_patient_clusters(df):

    # append cluster id's to df

    return df

def get_genome_clusters(df):


    # append cluster id's to df

    return df


def _get_reduced(df, reduction = 'filtered'):
    if reduction == 'filtered':
        df_reduced = get_filtered_genomes
    elif reduction == 'pca':
        df_reduced = get_principal_components
    elif reduction == 'lda':
        df_reduced = get_individual_components
    elif reduction == 'autoencoding':
        df_reduced = get_autoencoded_features
    elif reduction == 'mds':
        df_reduced = get_mds_features
    elif reduction == 'tsne':
        df_reduced = get_tsne_features
    elif reduction == 'rf': # random forest
        df_reduced = get_rf_features
    return df_reduced

def _graph_affinity_propagation(df):
    ## TO FINISH
    ## 

    return True

def _graph_community_detector(df):
    ## TO FINISH
    # maximize betweenness, modularity and group homogeneity

    return True



def _get_matrix(df, features = 'genomic', target = 'Treatment risk group in ALL10'): # type = ['genomic', ] 
    if(features =='genomic'):
        var_columns = df.columns[21:]# .values.tolist()
    elif(features =='patient'):
        var_columns = ["Age", "WhiteBloodCellcount", "Gender"]
        df[var_columns] = df[var_columns].fillna(0.0)

    train_idx = df[target].isin(["HR","MR","SR"])

    y = df[train_idx][target].map(lambda x: 0 if x in ["MR", "SR"] else 1).values
    df = df.drop(target, inplace = False, axis = 1)
    x = df.loc[train_idx,var_columns].values    
    return x,y

def _survival_matrix(df):
    valid = [0,1]
    gene_columns = df.columns[21:]

    target = "code_OS"

    df = df[df[target].isin(valid)]

    return df[gene_columns].values, df[target].values

def _preprocess(df, cohorts = ["cohort 1", "cohort 2", "JB", "IA", "ALL-10"], scaler = "standard"):
    gene_columns = df.columns[21:]
    if scaler == "standard":
        scaler = preprocessing.StandardScaler() # MinMaxScaler(), MaxAbsScaler(), RobustScaler(), QuantileTransformer(), Normalizer()
    elif scaler == "minmax":
        scaler = preprocessing.MinMaxScaler()
    elif scaler in ["normalizer", "normaliser"]:
        scaler = preprocessing.Normalizer()
    ch = df["array-batch"].isin(cohorts)
    df.loc[ch,gene_columns] = scaler.fit_transform(df.loc[ch,gene_columns])
    df = df[df["array-batch"].isin(cohorts)]
    return df

def _benchmark_classifier(model, x, y, splitter, seed, framework = 'sklearn'):
    splitter.random_state = seed
    pred = np.zeros(shape=y.shape)
    acc = np.zeros(shape=y.shape)
    coef = np.zeros(shape=(1, x.shape[1]))

    if framework == 'sklearn':
        for train_index, test_index in splitter.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index] 

            model[1].fit(x_train,y_train)
            pred_ = model[1].predict(x_test)
            pred[test_index] = pred_ 
            acc[test_index] = metrics.accuracy_score(y_test, pred_)
            # coef += model.coef_
    elif framework == 'custom_rvm':
        import rvm
        for train_index, test_index in splitter.split(x,y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]             
            
            model = rvm.rvm(x_train, y_train, noise = 0.01)
            model.iterateUntilConvergence()
            pred_  = np.round(np.reshape(np.dot(x_test, model.wInferred), newshape=[len(x_test),]));
            pred[test_index] = pred_
            acc[test_index] = metrics.accuracy_score(y_test, pred_)

    return pred, acc


def get_pca_transform(X, n_comp): # principal components, used for the classifiers
    pars = self.DIMENSION_REDUCTION_PARAMETERS['pca']
    Transform = decomposition.PCA(n_components = n_comp, **pars).fit(X)
    X_out = Transform.transform(X)
    return X_out, Transform


def get_lda_transform(X, y, n_comp): 
    pars = self.DIMENSION_REDUCTION_PARAMETERS['lda']
    Transform = discriminant_analysis.LinearDiscriminantAnalysis(n_components = n_comp, **pars).fit(X,y)
    X_out = Transform.transform(X)
    return X_out, Transform

def get_rbm_transform(X,y, n_comp):
    pars = self.DIMENSION_REDUCTION_PARAMETERS['rbm']
    Transform = neural_network.BernoulliRBM(n_components = n_comp, **pars).fit(X,y)
    X_out = Transform.transform(X)
    return X_out, Transform

def get_autoencoded_features(X,y):
    ## TO FINISH use Keras, or tensorflow

    # Transform is basically list of booleans
    return autoencoded_features, Transform 

def get_tsne_transform(X, y):
    #
    return X_out

def get_optics_clusters(X):
    #
    return cluster_ids


'''
def get_vector_characteristics():
    # 

    return True
'''

def get_filtered_genomes(x, filter_type = None):
    # low variance filter: minimum relative relative variance (var/mean)_i / (var/mean)_all 
    # 

    # low variance filter: minimum summed succesive (absolute) difference

    # 
    # Transform is basically list of booleans
    return True, Transform

 
def get_rf_weights(x, y, n):
    forest = ensemble.RandomForestClassifier(n_estimators=n, random_state=0, n_jobs=-1)
    forest.fit(x, y)
    importances = forest.feature_importances_
    return 

def get_et_weights(x, y, n):
    forest = ensemble.GradientBoostingClassifier(n_estimators=n, random_state=0, n_jobs=-1)
    forest.fit(x, y)
    importances = forest.feature_importances_
    return importances

def get_gbm_weights(x, y, n):
    forest = ensemble.ExtraTreesClassifier(n_estimators=n, random_state=0, n_jobs=-1)
    forest.fit(x, y)
    importances = forest.feature_importances_
    return importances

def get_ada_weights(x, y, n):
    forest = ensemble.AdaBoostClassifier(n_estimators=n, random_state=0, n_jobs=-1)
    forest.fit(x, y)
    importances = forest.feature_importances_
    return importances

def get_svm_weights(x, y):    
    forest = LinearSVC()
    forest.fit(x, y)
    importances = forest.coef_[0]
    return importances


def get_lr_weights(x, y, method = 'one-versus-all'):
    if (method == 'one-versus-all'):
        coef = []
    elif (method == 'all-versus-all'):
        coef = []
    return coef

def get_top_genes(x,y, method=None, n_max = 1000, n_comp = 1000, boruta = False):
    ''' Extract the genomes that are most relevant for the classification
    * method
    * extra-trees weights
    * RF weights
    * GBM weights
    * Adaboost weights
    * LR weights: all-versus-all 
    * LR weights: one-versus-all
    * SVM feature importance           

    # Boruta: https://pypi.python.org/pypi/Boruta/0.1.5 
    # http://scikit-learn.org/stable/modules/feature_selection.html
    '''   

    if (method in ['RF', 'SVM', 'ET', 'GBM', 'ADA']):
        coef = get_coef_sk(x, y, method, n_comp)
        topN = np.argsort(np.abs(coef))[::-1]
        #coef = np.reshape(coef, (coef.shape[1],))
        #ordering = np.argsort(np.abs(coef))
        #topN = ordering[-n:]
        if boruta:
            print("Not implemented yet")
    elif (method in ['LRall', 'LRone']):
        print("not done..")

    # if multiple methods are used, only keep overlapping genes
    gene_columns = self.DATA_merged.columns[21:].values
    top_genes = []
    for i in range(0, n_max):
        test = topN[i]
        top_genes.append({'probeset': gene_columns[test], 'rank_coeff': coef[test]})

    top_genes_df = pandas.DataFrame(top_genes)

    #coef_list = []
    #for coef in model._coef:
    #    coef_list.append(coef)

    return top_genes_df

    ########
    ## couple genomes to probesets