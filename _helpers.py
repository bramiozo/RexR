from sklearn import preprocessing, svm, tree, ensemble, naive_bayes, neural_network, model_selection, metrics
from sklearn import discriminant_analysis, decomposition, cross_decomposition
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix
from sklearn.cluster import AffinityPropagation
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.manifold import Isomap as ISO
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.model_selection import train_test_split, cross_val_score  
from sklearn.feature_selection import SelectFdr, SelectFpr
from sklearn.feature_selection import f_classif, chi2

import numpy as np
import pandas as pd

from collections import Counter 
from math import*

from scipy.spatial.distance import minkowski
from scipy.spatial.distance import cdist
from scipy import sparse
from scipy.stats import wilcoxon, mannwhitneyu


from decimal import Decimal
from time import time
import matplotlib.pyplot as plt
from itertools import cycle
from itertools import product

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import Callback

import lightgbm as lgb

from time import time
from functools import wraps

def timed(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        start = time()
        result = f(*args, **kwds)
        elapsed = time() - start
        print("{}.{} took {:4.2f} seconds to finish".format(f.__module__,
                                                     f.__name__, elapsed))
        return result
    return wrapper



class BatchLogger(Callback):
    def on_train_begin(self, epoch, logs={}):
        self.log_values = {}
        for k in self.params['metrics']:
            self.log_values[k] = []

    def on_epoch_end(self, batch, logs={}):
        for k in self.params['metrics']:
            if k in logs:
                self.log_values[k].append(logs[k])
    
    def get_values(self, metric_name, window):
        d =  pd.Series(self.log_values[metric_name])
        return d.rolling(window,center=False).mean()
BL = BatchLogger()

def plot_cm(ax, y_true, y_pred, classes, title, th=0.5, cmap=plt.cm.Blues):
    try:
        y_pred_labels = [np.round(l[1]).astype(int) for l in y_pred] # (y_pred>th).astype(int)
    except IndexError as e:
        y_pred_labels = [np.round(l).astype(int) for l in y_pred]

    cm = confusion_matrix(y_true, y_pred_labels)
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

def plot_auc(ax, y_train, y_train_pred, y_test, y_test_pred, th=0.5):

    try:
        y_train_pred_labels = [np.round(l[1]).astype(int) for l in y_train_pred]#(y_train_pred>th).astype(int)
        y_test_pred_labels  = [np.round(l[1]).astype(int) for l in y_test_pred]#(y_test_pred>th).astype(int)
        y_train_pred = [l[1] for l in y_train_pred]
        y_test_pred = [l[1] for l in y_test_pred]
    except IndexError as e:
        y_train_pred_labels = [np.round(l).astype(int) for l in y_train_pred]#(y_train_pred>th).astype(int)
        y_test_pred_labels  = [np.round(l).astype(int) for l in y_test_pred]#(y_test_pred>th).astype(int)     
        y_train_pred = [l for l in y_train_pred]
        y_test_pred = [l for l in y_test_pred]


    fpr_train, tpr_train, _ = metrics.roc_curve(y_train,y_train_pred)
    roc_auc_train = metrics.auc(fpr_train, tpr_train)
    acc_train = metrics.accuracy_score(y_train, y_train_pred_labels)

    fpr_test, tpr_test, _ = metrics.roc_curve(y_test,y_test_pred)
    roc_auc_test = metrics.auc(fpr_test, tpr_test)
    acc_test = metrics.accuracy_score(y_test, y_test_pred_labels)

    ax.plot(fpr_train, tpr_train)
    ax.plot(fpr_test, tpr_test)

    ax.plot([0, 1], [0, 1], 'k--')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curve')
    
    train_text = 'train acc = {:.3f}, auc = {:.2f}'.format(acc_train, roc_auc_train)
    test_text = 'test acc = {:.3f}, auc = {:.2f}'.format(acc_test, roc_auc_test)
    ax.legend([train_text, test_text])


def _group_patients(df, method = 'first', Rclass = None): # method = ['first', 'mean', 'median', 'min', 'max']
    # 1. clean probesets 
    # 2. reduce multiple probesets per gene to one  
    if(Rclass.SET_NAME == 'ALL_10'):
        df_1 = df[df.columns[:21]].groupby(Rclass.MODEL_PARAMETERS['ID']).apply(lambda g: g.iloc[0])
        col_list = df.columns[21:].values.tolist()
        col_list.append(Rclass.MODEL_PARAMETERS['ID'])
        if(method == 'first'):
            df_2 = df[col_list].groupby(Rclass.MODEL_PARAMETERS['ID']).apply(lambda g: g.iloc[0])
        elif(method == 'mean'):
            df_2 = df[col_list].groupby(Rclass.MODEL_PARAMETERS['ID']).mean()
        elif(method == 'median'):
            df_2 = df[col_list].groupby(Rclass.MODEL_PARAMETERS['ID']).median()
        elif(method == 'min'):
            df_2 = df[col_list].groupby(Rclass.MODEL_PARAMETERS['ID']).min()
        elif(method == 'max'):
            df_2 = df[col_list].groupby(Rclass.MODEL_PARAMETERS['ID']).max()  
        dfinal = df_1.merge(df_2, left_index = True, right_index = True)
        return dfinal
    else:
        return df


def patient_similarity(patient_matrix, sim_type = 'cosine', minkowski_dim = None, normalised = True, inflation = 1):
    """ Function to get similarity measures between patients
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
    """
    
    if len(patient_matrix<1000): 
        if (sim_type in ['cosine', 'manhattan', 'euclidian', 'minkowski', 'braycurtis', 'canberra', 'chebyshev', 'dice','hamming', 'jaccard', 'kulsinski', 'mahalanobis', 
                    'matching', 'rogerstanimoto', 'russellrao', 'seuclidean', '‘sokalmichener', 'sokalsneath']):
            A = np.array(patient_matrix)
        else:
            A = None
        VarList = patient_matrix.T.keys()
        if(sim_type == 'cosine'):        
            similarities = cdist(A, A, metric = 'cosine')
            patient_similarity = pd.DataFrame(similarities, index = VarList, columns = VarList)
        elif(sim_type == 'manhattan'):
            similarities = cdist(A, A, metric = 'cityblock')
            patient_similarity = pd.DataFrame(similarities, index = VarList, columns = VarList)        
        elif(sim_type == 'euclidian'):
            similarities = cdist(A, A, metric = 'euclidean')
            patient_similarity = pd.DataFrame(similarities, index = VarList, columns = VarList)
        elif(sim_type == 'minkowski'):
            if (minkowski_dim == None):
                print("No Minkowski dimension given! Assuming minkowski dim is 3")
                minkowski_dim = 3
            similarities = cdist(A, A, metric = 'minkowski', p = minkowski_dim)
            patient_similarity = pd.DataFrame(similarities, index = VarList, columns = VarList)       
        elif(sim_type in ['kendall', 'spearman', 'pearson']):
            patient_similarity = 1-patient_matrix.T.astype('float64').corr(method = sim_type)    
        elif(sim_type in ['braycurtis', 'canberra', 'chebyshev', 'dice','hamming', 'jaccard', 'kulsinski', 'mahalanobis', 
                    'matching', 'rogerstanimoto', 'russellrao', 'seuclidean', '‘sokalmichener', 'sokalsneath']):
            similarities = cdist(A, A, metric = sim_type)
            patient_similarity = pd.DataFrame(similarities, index = VarList, columns = VarList)       
        ###
        if inflation > 1:
            patient_similarity = patient_similarity**inflation
        ###                                                               
        if normalised == True: # ! IMPROVE, not memory efficient
            patient_similarity = (patient_similarity - min(patient_similarity.min()))/(max(patient_similarity.max())-min(patient_similarity.min()))

        patient_similarity = 1 - patient_similarity
    else:
        raise ValueError('More than 1000 samples not supported at this moment')
    ###
    return patient_similarity


def _get_matrix(df, features = 'genomic', target = 'Treatment_risk_group_in_ALL10', Rclass = None): # type = ['genomic', ] 
    if(Rclass.SET_NAME=='ALL_10'):
        if target == 'Treatment_risk_group_in_ALL10':
            if(features =='genomic'):
                var_columns = df.columns[21:]# .values.tolist()
            elif(features =='patient'):
                var_columns = ["Age", "WhiteBloodCellcount", "Gender", 
                               "mutations_NOTCH_pathway", "mutations_IL7R_pathway",
                               "mutations_PTEN_AKT_pathway", "code_OS"]
                df[var_columns] = df[var_columns].fillna(0.0)

            train_idx = df[target].isin(["HR","MR","SR"])

            y = df[train_idx][target].map(lambda x: 0 if x in ["MR", "SR"] else 1).values
            df = df.drop(target, inplace = False, axis = 1)
            x = df.loc[train_idx,var_columns].values

            return x, y
        elif target == 'code_OS':
            valid = [0,1]
            gene_columns = df.columns[21:]
            target = "code_OS"
            df = df[df[target].isin(valid)]

            return df[gene_columns].values, df[target].values           
    elif(Rclass.SET_NAME =='MELA'):
        y = df[:]['target'].map(lambda x: 0 if x =="Genotype: primary tumor" else 1).values
        df = df.drop(target, inplace = False, axis = 1)
        x = df.loc[:, (df.columns!='target') & (df.columns!='ID')].values            
        return x, y

def _add_noise(x, noise_level = 0.01, noise_type = 'absolute'):
    # absolute noise addition: x_ij +- noise_level
    if noise_type == 'absolute':
        x = np.random.normal(x, scale = noise_level)
    # relative noise addition: x_ij* (1 + noise_level)
    elif noise_type == 'relative':
        noise_level = noise_level * np.mean(x)
        x = np.random.normal(x, scale = noise_level)
    return x

def _survival_matrix(df):
    valid = [0,1]
    gene_columns = df.columns[21:]
    target = "code_OS"
    df = df[df[target].isin(valid)]

    return df[gene_columns].values, df[target].values

def _preprocess(df, cohorts = ["cohort 1", "cohort 2", "JB", "IA", "ALL-10"], scaler = "standard", Rclass = None):
    if(Rclass.SET_NAME == 'ALL_10'):
        gene_columns = df.columns[21:]
        # MinMaxScaler(), MaxAbsScaler(), RobustScaler(), QuantileTransformer(), Normalizer()
        if scaler == "standard":
            scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
        elif scaler == "minmax":
            scaler = preprocessing.MinMaxScaler()
        elif scaler == "robust":
            scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0), 
                                                    with_scaling=True, with_centering=True)
        elif scaler in ["normalizer", "normaliser"]:
            scaler = preprocessing.Normalizer()

        if Rclass.PIPELINE_PARAMETERS['pre_processing']['bias_removal'] == True:
            # Assume for now we simply perform cohort-wise normalisation..
            print("- "*30, 'Removing cohort biases')
            for cohort in cohorts:
                ch = df["array-batch"] == cohort
                df.loc[ch,gene_columns] = scaler.fit_transform(df.loc[ch,gene_columns])
                if Rclass.PIPELINE_PARAMETERS['scaler']['maxabs']==True:
                    print("- "*30, 'Apply maxabs scaling')
                    scaler = preprocessing.MaxAbsScaler()
                    df.loc[ch,gene_columns] = scaler.fit_transform(df.loc[ch,gene_columns])
        else:
            ch = df["array-batch"].isin(cohorts)
            df.loc[ch,gene_columns] = scaler.fit_transform(df.loc[ch,gene_columns])
            if Rclass.PIPELINE_PARAMETERS['scaler']['maxabs']==True:
                print("- "*30, 'Apply maxabs scaling')
                scaler = preprocessing.MaxAbsScaler()
                df.loc[ch,gene_columns] = scaler.fit_transform(df.loc[ch,gene_columns])
            

        df = df[df["array-batch"].isin(cohorts)]
        return df
    elif(Rclass.SET_NAME == 'MELA'):
        gene_columns = df.loc[:, (df.columns!='target') & (df.columns!='ID')].columns
        #df[gene_columns] = df[gene_columns].apply(pd.to_numeric)
        if scaler == "standard":
            scaler = preprocessing.StandardScaler() # MinMaxScaler(), MaxAbsScaler(), RobustScaler(), QuantileTransformer(), Normalizer()
        elif scaler == "minmax":
            scaler = preprocessing.MinMaxScaler()
        elif scaler in ["normalizer", "normaliser"]:
            scaler = preprocessing.Normalizer()
        df.loc[:,gene_columns] = scaler.fit_transform(df.loc[:,gene_columns])
        if Rclass.PIPELINE_PARAMETERS['scaler']['maxabs']==True:
            print("- "*30, 'Apply maxabs scaling')
            scaler = preprocessing.MaxAbsScaler()
            df.loc[ch,gene_columns] = scaler.fit_transform(df.loc[ch,gene_columns])        
        return df

def _benchmark_classifier(model, x, y, splitter, seed, framework = 'sklearn', Rclass = None):
    splitter.random_state = seed
    pred = np.zeros(shape=y.shape)
    acc = np.zeros(shape=y.shape)
    coef = np.zeros(shape=(1, x.shape[1]))
    threshold = 0.5
    if framework == 'sklearn':
        for train_index, test_index in splitter.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model[1].fit(x_train, y_train)
#           pred_test = model[1].predict_proba(x_test) # (model[1].predict_proba(x_test)>threshold).astype(int)
            pred_test_ = model[1].predict(x_test) #[np.round(l[1]).astype(int) for l in pred_test]
            pred[test_index] =  pred_test_ #np.round(pred_test)[0]
            acc[test_index] = metrics.accuracy_score(y_test, pred_test_)
            # coef += model.coef_            
        ######################################################
        ##### For last split, show confusion matrix and ROC ##
        ######################################################
        #X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=0.2, random_state=42)
        if Rclass.VIZ == True:
            pred_train = model[1].predict_proba(x_train)
            fig,ax = plt.subplots(1,3)
            fig.set_size_inches(15,5)
            plot_cm(ax[0],  y_train, pred_train, [0,1], 'Confusion matrix (TRAIN)', threshold)
            plot_cm(ax[1],  y_test, pred_test,   [0,1], 'Confusion matrix (TEST)', threshold)
            plot_auc(ax[2], y_train, pred_train, y_test, pred_test, threshold)
            plt.tight_layout()
            plt.show()


    elif framework == 'custom_rvm':
        import rvm
        for train_index, test_index in splitter.split(x,y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]             
            
            model = rvm.rvm(x_train, y_train, noise = 0.01)
            model.iterateUntilConvergence()
            pred_test   = np.reshape(np.dot(x_test, model.wInferred), newshape=[len(x_test),])/2+0.5
            pred_test_  = np.round(pred_test);
            pred[test_index] = pred_test_
            acc[test_index] = metrics.accuracy_score(y_test, pred_test_)

        ######################################################
        ##### For last split, show confusion matrix and ROC ##
        ######################################################
        #X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=0.2, random_state=42)
        if Rclass.VIZ == True:
            pred_train = np.reshape(np.dot(x_train, model.wInferred), newshape=[len(x_train),])/2+0.5
            fig,ax = plt.subplots(1,3)
            fig.set_size_inches(15,5)
            plot_cm(ax[0],  y_train, pred_train, [0,1], 'Confusion matrix (TRAIN)', threshold)
            plot_cm(ax[1],  y_test, pred_test,   [0,1], 'Confusion matrix (TEST)', threshold)
            plot_auc(ax[2], y_train, pred_train, y_test, pred_test, threshold)
            plt.tight_layout()
            plt.show()   
    elif framework == 'keras':
        for train_index, test_index in splitter.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]             
            model[1].fit(x_train, y_train, batch_size = 10, epochs = 5, verbose = 1, 
                callbacks=[BL], validation_data=(np.array(x_test), np.array(y_test))) 
            #score = model[1].evaluate(np.array(x_test), np.array(y_test), verbose=0)
            #print('Test log loss:', score[0])
            #print('Test accuracy:', score[1])
            pred_test = model[1].predict(x_test)[:,0]
            pred_test_ = np.round(pred_test)
            pred[test_index] = pred_test_
            acc[test_index] = metrics.accuracy_score(y_test, pred_test_)

        ######################################################
        ##### For last split, show confusion matrix and ROC ##
        ######################################################     
        # https://github.com/natbusa/deepcredit/blob/master/default-prediction.ipynb   
        if Rclass.VIZ == True:
            plt.figure(figsize=(15,5))
            plt.subplot(1, 2, 1)
            plt.title('loss, per batch')
            plt.plot(BL.get_values('loss',1), 'b-', label='train');
            plt.plot(BL.get_values('val_loss',1), 'r-', label='test');
            plt.legend()
            #
            plt.subplot(1, 2, 2)
            plt.title('accuracy, per batch')
            plt.plot(BL.get_values('acc',1), 'b-', label='train');
            plt.plot(BL.get_values('val_acc',1), 'r-', label='test');
            plt.legend()
            plt.show() 

            y_train_pred = model[1].predict_on_batch(np.array(x_train))[:,0]
            y_test_pred = model[1].predict_on_batch(np.array(x_test))[:,0]

            fig,ax = plt.subplots(1,3)
            fig.set_size_inches(15,5)

            plot_cm(ax[0], y_train, y_train_pred, [0,1], 'Confusion matrix (TRAIN)')
            plot_cm(ax[1], y_test, y_test_pred, [0,1], 'Confusion matrix (TEST)')

            plot_auc(ax[2], y_train, y_train_pred, y_test, y_test_pred)
                
            plt.tight_layout()
            plt.show()
    return pred, acc


def get_dim_reduction(X, y = None, n_comp = 1000, method = 'pca', Rclass = None):
    #print("\ "*15, 'Dimension reduction, current matrix shape is {}, N is {}'.format(X.shape, n_comp))
    if(method.lower() == 'pca'):
        Rclass.DIMENSION_REDUCTION_PARAMETERS['pca']['n_components'] = n_comp
        pars = Rclass.DIMENSION_REDUCTION_PARAMETERS['pca']
        Transform = decomposition.PCA(**pars).fit(X)
    elif(method.lower() == 'lda'):
        pars = Rclass.DIMENSION_REDUCTION_PARAMETERS['lda']
        Rclass.DIMENSION_REDUCTION_PARAMETERS['lda']['n_components'] = n_comp
        Transform = discriminant_analysis.LinearDiscriminantAnalysis(**pars).fit(X,y)
    elif(method.lower() == 'pls'):
        pars = Rclass.DIMENSION_REDUCTION_PARAMETERS['pls']
        Transform = cross_decomposition.PLSRegression(n_components = n_comp, **pars).fit(X, y)
    elif(method.lower() == 'rbm'):
        pars = Rclass.DIMENSION_REDUCTION_PARAMETERS['rbm']
        Transform = neural_network.BernoulliRBM(n_components = n_comp, **pars).fit(X,y)
    elif(method.lower() == 'lle'):
        pars = Rclass.DIMENSION_REDUCTION_PARAMETERS['lle']
        Transform = LLE(**pars).fit(X)
    elif(method.lower() == 'isomap'):
        pars = Rclass.DIMENSION_REDUCTION_PARAMETERS['isomap']
        Transform = ISO(**pars).fit(X)
    elif(method.lower() == 'mds'):
        pars = Rclass.DIMENSION_REDUCTION_PARAMETERS['mds']
        Transform = MDS(**pars).fit(X)
    elif(method.lower() == 't-sne'):
        pars = Rclass.DIMENSION_REDUCTION_PARAMETERS['t-sne']
        Transform = TSNE(**pars).fit(X)
    elif(method.lower() == 'sae'):
        #https://github.com/fchollet/keras/issues/358
        return True
    #elif(method.lower() == 'genome_variance'):
    #    X_out, Transform = get_filtered_genomes(x_, filter_type = None)
    #    return X_out, Transform
    else:
        raise ValueError 

    X_out = Transform.transform(X)
    #print("/ "*15, 'Reduced matrix shape is {}'.format(X_out.shape))
    return X_out, Transform

def get_top_genes(MODELS=None, n_max = 1000, sort_by = 'MEAN', RexR=None):
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
    top_genomes_weights = pd.DataFrame()
    for idx, mod in enumerate(MODELS):
        if(mod['method'].lower() in ['rf', 'et', 'randomforest', 'gbm', 
                                    'adaboost', 'extratrees', 'xgb', 'ada', 'lgbm']): # RF, ET, GBM, ADA
            try:
                method_name = str(idx)+'_'+mod['method']
                top_genomes_weights[method_name]=mod['model'].feature_importances_
                # column normalise
                top_genomes_weights[method_name] = top_genomes_weights[method_name]/\
                                                top_genomes_weights[method_name].max()
            except AttributeError as e:
                print("model {} does not contain feature importances: {}".format(mod['method'] , e))
    # if multiple methods are used, only keep overlapping genes

    if(RexR.SET_NAME == 'MELA'):        
        if(RexR.PREP_DEL is not None):
            feat_sel = RexR.DATA_merged_processed.columns[RexR.PREP_DEL]
            top_genomes_weights.index = RexR.DATA_merged_processed.drop(feat_sel, axis=1)\
                                                                  .drop(['target', 'ID'], axis=1)\
                                                                  .columns
        else:
            top_genomes_weights.index = RexR.DATA_merged_processed.drop(['target', 'ID'], axis=1)\
                                                                  .columns          
        
    elif(RexR.SET_NAME == 'ALL_10'):
        pre_cols = RexR.DATA_merged_processed.columns[:21]
        if(RexR.PREP_DEL is not None):
            feat_sel = RexR.DATA_merged_processed.columns[21:][RexR.PREP_DEL]
            top_genomes_weights.index = RexR.DATA_merged_processed.drop(pre_cols, axis=1)\
                                                                  .drop(feat_sel, axis=1)\
                                                                  .columns
        else:
            top_genomes_weights.index = RexR.DATA_merged_processed.drop(pre_cols, axis=1)\
                                                                  .columns            
	        

    top_genomes_weights['MEAN'] = top_genomes_weights.mean(axis=1)
    top_genomes_weights['MEDIAN'] = top_genomes_weights.median(axis=1)
    top_genomes_index = top_genomes_weights.index
    top_genomes_weights = top_genomes_weights.sort_values(by=sort_by, ascending=False)[:n_max]
           
    ### Coefficients
    top_genomes_coeffs = pd.DataFrame()
    coeffs_check = False
    for mod in MODELS:
        if(mod['method'] in ['LR', 'SVM']):
            coeffs_check = True
            method_name = str(idx)+'_'+mod['method']
            top_genomes_coeffs[method_name] = mod['model'].coef_[0,:]
            top_genomes_coeffs[method_name] = top_genomes_coeffs[method_name]/\
                                            top_genomes_coeffs[method_name].max() #\
                                                                   #  -top_genomes[mod['method']].min())
                                                                     #+numpy.abs(top_genomes[mod['method']].min())
    if coeffs_check:
        top_genomes_coeffs.index = top_genomes_index
        # if(RexR.SET_NAME == 'MELA'):            
        #     top_genomes_coeffs.index = top_genomes_weights.index
        # elif(RexR.SET_NAME == 'ALL_10'):
        #     pre_cols = RexR.DATA_merged_processed.columns[:21]
        #     if(RexR.PREP_DEL is not None):
        #         feat_sel = RexR.DATA_merged_processed.columns[21:][RexR.PREP_DEL]
        #         top_genomes_coeffs.index = RexR.DATA_merged_processed.drop(pre_cols, axis=1)\
        #                                                              .drop(feat_sel, axis=1)\
        #                                                              .columns
        #     else:
        #         top_genomes_coeffs.index = RexR.DATA_merged_processed.drop(pre_cols, axis=1)\
        #                                                               .columns     
        #top_genomes['ALL'] = top_genomes.sum(axis=1)
        top_genomes_coeffs['MEAN'] = top_genomes_coeffs.mean(axis=1)
        top_genomes_coeffs['MEDIAN'] = top_genomes_coeffs.median(axis=1)
        top_genomes_coeffs = top_genomes_coeffs.sort_values(by=sort_by, ascending=False)[:]    

        return top_genomes_weights, top_genomes_coeffs[-int(n_max/2):].append(top_genomes_coeffs[:int(n_max/2)])\
                                                                  .sort_values(by=sort_by)
    else:
        return top_genomes_weights, top_genomes_coeffs



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

def _cohort_correction(df):
    ## TO FINISH
    # correct for bias introduced by measurements
    # check for means in measurement groups, for similar patients (use groups from affinity propagation?)
    # assumptions: patients sampled over different groups in a stratified manner

    return True

def gene_map(genes = None, color_scheme = None, map_type = None, gene_sim = None):
    return True


def _graph_affinity_propagation(df):
    ## TO FINISH
    ## 

    return True

def _graph_markov_clustering(df):
    return True

def _graph_community_detector(df, method = "SBM"):
    ## TO FINISH
    # method: SBM, Louvain, AP
    # ap :http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.399.6701&rep=rep1&type=pdf, change preference and check modularity
    # maximize betweenness, modularity and group homogeneity, minimize conductance
    # https://www.youtube.com/watch?v=jIS5pZ8doH8
    return True

@timed
def _probeset_mapper(probeset_type = 'HT_HG-U133', 
                     mapping_file = None,
                     probeset_col = 'Probe Set ID', 
                     description_list = ['Pathway','Gene Title', 'Gene Symbol', 'Chromosomal Location'], 
                     probe_list = []):
    # if mapping_file is empty one will be taken from the predefined list based
    # on the probeset_type
    # description_list: is a list of columns to append to the probeset id in the dictionary
    # probe_list: is a list of probesets to map
    if mapping_file == None:
        if(probeset_type == 'HT_HG-U133'):
            mapping_file = '_data/genomic_data/mapping-data/HT_HG-U133_Plus_PM.na35.annot.txt'
        elif(probeset_type == 'HG-U133'):
            mapping_file = '_data/genomic_data/mapping-data/HG-U133_Plus_2.na36annot.txt'

    ColSelect = description_list.append(probeset_col)
    mapping_data = pd.read_csv(mapping_file, usecols = ColSelect, sep='\t')

    probemap = {}
    for probeset_id in probe_list:
        try:
            probemap[probeset_id] = dict(zip(mapping_data.loc[mapping_data[probeset_col]==probeset_id].keys().tolist(), 
                                         mapping_data.loc[mapping_data[probeset_col]==probeset_id].values[0].tolist()))
        except Exception as e:
            print("Problem with mapping the probes onto the genomes : {}".format(e))


    return probemap

def get_genome_similarity(df, reduction = 'filtered', max_dim = 10000):
    ## TO FINISH, low-dim, high number of vectors
    ## need to create sparse representation otherwise we end up with a 55.000 x 55.000 matrix
    df_reduced = _get_reduced(df, reduce_type= reduction)

    return True

def get_patient_clusters(df, method="AP"):
    # methods: AP, MCL, SOG
    # append cluster id's to df

    return df

def get_genome_clusters(df):


    # append cluster id's to df

    return df

def get_vector_characteristics():
    # 

    return True


class fs_mannwhitney():
    pvalue = 0.01
    p_values = None

    def __init__(self, pvalue = 0.01):
        self.pvalue = pvalue

    def apply_test(self, pos, neg, column):
        _, p_value = mannwhitneyu(pos[:,column], neg[:,column], alternative="less")
        return p_value

    def fit(self, x, y):
        zero_idx = np.where(y == 0)[0]
        one_idx = np.where(y == 1)[0]
        pos_samples = x[one_idx]
        neg_samples = x[zero_idx]                
        self.p_values = np.array(list(map(lambda c: 
            self.apply_test(pos_samples, neg_samples, c), range(0,x.shape[1]))))
        return self

    def transform(self, x):
        not_signif = self.p_values<self.pvalue
        to_delete = [idx for idx, item in enumerate(not_signif) if item == False]
        return np.delete(x, to_delete, axis = 1), to_delete

def get_filtered_genomes(x, y, Rclass = None):
    try:
        alpha = Rclass.PIPELINE_PARAMETERS['feature_selection']['pvalue']
        filter_type = Rclass.PIPELINE_PARAMETERS['feature_selection']['method']
        F_function =  Rclass.PIPELINE_PARAMETERS['feature_selection']['score_function']
    except Exception as e:
        print("Exception with {} handling the function get_filtered_genomes".format(e))
        alpha = 0.05
        filter_type = 'FDR'
        F_function = 'ANOVA'

    
    
    #  Mann-Whitney, between classes
    # scipy.stats.mannwhitneyu, https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
    if filter_type == 'mannwhitney':
        Selector = fs_mannwhitney(pvalue = alpha).fit(x,y)
        x_out = Selector.transform(x)[0]
    elif filter_type == 'FDR':
    # Use FDR with a number of different statistical measures:
    # f_classif, chi2, 
        FDR_function = f_classif if F_function == 'ANOVA' else eval(F_function)
        FDR = SelectFdr(alpha = alpha, score_func = FDR_function) #f_classif, chi2
        Selector = FDR.fit(x, y)
        x_out = FDR.transform(x)
    elif filter_type == 'FPR':
        FPR_function = f_classif if F_function == 'ANOVA' else eval(F_function)
        FPR = SelectFpr(alpha = alpha, score_func = FPR_function) #f_classif, chi2
        Selector = FPR.fit(x, y)
        x_out = FPR.transform(x)        

    # 4. Wilcoxon signed-rank, between classes
    # scipy.stats.wilcoxon, https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html


    # 5. Chi-Square, between classes
    # scipy.stats.chisquare, https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html

    # 6. t-test independent, between classes
    # scipy.stats.ttest_ind, https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.stats.ttest_ind.html

    # 7. t-test related, between classes
    # scipy.stats.ttest_rel

    # 8. remove collinearity of feature vectors within class set (leave one)

    # 9. remove collinearity of feature vectors between classes (remove both)

    # Transform is basically list of booleans

    return x_out, Selector

def _get_filtered_matrix(x, filter_method = 'FDR', loops = 10, noise_level = 0.01, Rclass = None):
    '''
        filter_method: FDR, MW-U (univariate) 
    '''

    # normalise?

    filter_res = []
    # loop
        # add noise

        # filter

        # 


    return x_out, filter_res

def rotation_norm(x, y, norm):   
    rot_vector = numpy.cross(a = x, b = y)
    norm_of_rotation = numpy.linalg.norm(rot_vector, ord = norm)
    ## example
    #reduced = dim_reduction(Transposed, dims = 3) # reduce to 3 or 2 dimensions
    #rotation_norm(reduced['9827_corr2.CEL'], reduced['9928_corr2.CEL'], 2)
    return norm_of_rotation


def get_difference_markers():
    # given two different groups from unsupervised clustering, 
    # find most important genome expressions
    # by treating it as a binary classification problem
    return True;


def get_outliers(X, method = "sos"):
    # sos, t-sne --> lof; t-sne --> dbscan, t-sne with pij = sqrt(pi|j*pj|i)

    return True;


def get_lr_weights(x, y, method = 'one-versus-all'):
    if (method == 'one-versus-all'):
        coef = []
    elif (method == 'all-versus-all'):
        coef = []
    return coef
