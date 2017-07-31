from sklearn import preprocessing, svm, tree, ensemble, naive_bayes, neural_network, model_selection, metrics
from sklearn import discriminant_analysis, decomposition
import numpy as np
import pandas as pd
from collections import Counter 


def _group_patients(df, method = 'first'): # method = ['first', 'average', 'median', 'min', 'max']
    # 1. clean probesets 
    # 2. reduce multiple probesets per gene to one  
    if(method == 'first'):
        df = df.groupby("labnr patient").apply(lambda g: g.iloc[0])
    elif(method == 'average'):
        df = df.groupby("labnr patient").mean()
    elif(method == 'median'):
        df = df.groupby("labnr patient").median()
    elif(method == 'min'):
        df = df.groupby("labnr patient").min()
    elif(method == 'max'):
        df = df.groupby("labnr patient").max()

    return df 

def _cohort_correction(df):
    # correct for bias introduced by measurements
    return True

def _get_matrix(df, type = 'genomic', target = 'Treatment risk group in ALL10'): # type = ['genomic', ] 
    if(type=='genomic'):
        var_columns = df.columns[21:]
    elif(type=='patient'):
        var_columns = ["Age", "WhiteBloodCellcount", "Gender"]
        df[var_columns] = df[var_columns].fillna(0.0)

    train_idx = df[target].isin(["HR","MR","SR"])

    y = df[train_idx][target].map(lambda x: 0 if x in ["MR", "SR"] else 1).values
    x = df.loc[train_idx,var_columns].values

    return x,y

def _survival_matrix(df):
    valid = [0,1]
    gene_columns = df.columns[21:]

    target = "code_OS"

    df = df[df[target].isin(valid)]

    return df[gene_columns].values, df[target].values

def _preprocess(df, cohorts = ["cohort 1", "cohort 2", "JB", "IA", "ALL-10"]):
    gene_columns = df.columns[21:]
    scaler = preprocessing.StandardScaler()
    ch = df["array-batch"].isin(cohorts)
    df.loc[ch,gene_columns] = scaler.fit_transform(df.loc[ch,gene_columns])
    df = df[df["array-batch"].isin(cohorts)]
    return df

def _benchmark_classifier(model, x, y, splitter, seed):
    splitter.random_state = seed
    pred = np.zeros(shape=y.shape)
    acc = np.zeros(shape=y.shape)
    coef = np.zeros(shape=(1, x.shape[1]))

    for train_index, test_index in splitter.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index] 

        model[1].fit(x_train,y_train)
        pred_ = model[1].predict(x_test)
        pred[test_index] = pred_ 
        acc[test_index] = metrics.accuracy_score(y_test, pred_)
        # coef += model.coef_
        

    return pred, acc


def get_principal_components(X, n_comp):
    X_out = decomposition.PCA(n_components=n_comp, 
                            copy=True, whiten=False, 
                            svd_solver='auto', 
                            tol=0.0, iterated_power='auto', 
                            random_state=None).fit_transform(X)

    return X_out

def get_linear_discriminant_analysis(X, y):


    return lda_transformed

def get_quadrant_discriminant_analysis(X, y):

    return qda_transformed

'''
def get_vector_characteristics():
    # 

    return True

def get_genome_variation(x, min_norm_var = 0.2):
    # get variation per genome over all patients -> make sure that the classes are evenly distributed for this
    var_vector = numpy.ndarray.var(x)
    keep = var_vector > min_norm_var
    
    return x
''' 

