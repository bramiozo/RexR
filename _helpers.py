from sklearn import preprocessing, svm, tree, ensemble, naive_bayes, neural_network, model_selection, metrics
from sklearn import discriminant_analysis, decomposition, cross_decomposition
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix
from sklearn.cluster import AffinityPropagation

import numpy as np
import pandas as pd

from collections import Counter 
from math import*

from scipy.spatial.distance import minkowski
from scipy.spatial.distance import cdist
from scipy import sparse

from decimal import Decimal
from time import time
import matplotlib.pyplot as plt
from itertools import cycle
from itertools import product

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import Callback

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

def _cohort_correction(df):
    ## TO FINISH
    # correct for bias introduced by measurements
    # check for means in measurement groups, for similar patients (use groups from affinity propagation?)
    # assumptions: patients sampled over different groups in a stratified manner



    return True

def gene_map(genes = None, color_scheme = None, map_type = None, gene_sim = None):


    return True

def patient_similarity(patient_matrix, sim_type = 'cosine', minkowski_dim = None, normalised = True, inflation = 1):
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

def _graph_markov_clustering(df):

    return True

def _graph_community_detector(df, method = "SBM"):
    ## TO FINISH
    # method: SBM, Louvain, AP
    # ap :http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.399.6701&rep=rep1&type=pdf, change preference and check modularity
    # maximize betweenness, modularity and group homogeneity, minimize conductance
    # https://www.youtube.com/watch?v=jIS5pZ8doH8

    return True



def _get_matrix(df, features = 'genomic', target = 'Treatment_risk_group_in_ALL10', Rclass = None): # type = ['genomic', ] 
    if(Rclass.SET_NAME=='ALL_10'):
        if(features =='genomic'):
            var_columns = df.columns[21:]# .values.tolist()
        elif(features =='patient'):
            var_columns = ["Age", "WhiteBloodCellcount", "Gender"]
            df[var_columns] = df[var_columns].fillna(0.0)

        train_idx = df[target].isin(["HR","MR","SR"])

        y = df[train_idx][target].map(lambda x: 0 if x in ["MR", "SR"] else 1).values
        df = df.drop(target, inplace = False, axis = 1)
        x = df.loc[train_idx,var_columns].values    
    elif(Rclass.SET_NAME =='MELA'):
        y = df[:]['target'].map(lambda x: 0 if x =="Genotype: primary tumor" else 1).values
        df = df.drop(target, inplace = False, axis = 1)
        x = df.loc[:, (df.columns!='target') & (df.columns!='ID')].values            

    return x,y

def _survival_matrix(df):
    valid = [0,1]
    gene_columns = df.columns[21:]
    target = "code_OS"
    df = df[df[target].isin(valid)]

    return df[gene_columns].values, df[target].values

def _preprocess(df, cohorts = ["cohort 1", "cohort 2", "JB", "IA", "ALL-10"], scaler = "standard", Rclass = None):
    if(Rclass.SET_NAME == 'ALL_10'):
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
            model[1].fit(x_train,y_train)
            pred_test = model[1].predict_proba(x_test) # (model[1].predict_proba(x_test)>threshold).astype(int)
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


def get_pca_transform(X, n_comp, RexR): # principal components, used for the classifiers
    pars = RexR.DIMENSION_REDUCTION_PARAMETERS['pca']
    Transform = decomposition.PCA(n_components = n_comp, **pars).fit(X)
    X_out = Transform.transform(X)
    return X_out, Transform

def get_pls_transform(X,y, n_comp, RexR):
    pars = RexR.DIMENSION_REDUCTION_PARAMETERS['pls']
    Transform = cross_decomposition.PLSRegression(n_components = n_comp, **pars).fit(X, y)
    X_out = Transform.transform(X)
    return X_out, Transform

def get_ica_transform(X, n_comp, RexR): # individual component analysis
    pars = RexR.DIMENSION_REDUCTION_PARAMETERS['ica']
    Transform = decomposition.FastICA(n_components = n_comp, **pars).fit(X)
    X_out = Transform.transform(X)
    return X_out, Transform  

def get_lda_transform(X, y, n_comp, RexR): 
    pars = RexR.DIMENSION_REDUCTION_PARAMETERS['lda']
    Transform = discriminant_analysis.LinearDiscriminantAnalysis(n_components = n_comp, **pars).fit(X,y)
    X_out = Transform.transform(X)
    return X_out, Transform

def get_rbm_transform(X,y, n_comp, RexR):
    pars = RexR.DIMENSION_REDUCTION_PARAMETERS['rbm']
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

def get_mds_transform(X, y):
    return True


def get_vector_characteristics():
    # 

    return True


def get_filtered_genomes(x, filter_type = None):
    # 1. low variance filter: minimum relative relative variance (var/mean)_i / (var/mean)_all 

    # 2. low variance filter: minimum summed succesive (absolute) differences

    # 3. Wilcoxon-Mann-Whitney, between classes
    # scipy.stats.mannwhitneyu, https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html

    # 4. Wilcoxon signed-rank, between classes
    # scipy.stats.wilcoxon, https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html

    # 4. Chi-Square, between classes
    # scipy.stats.chisquare, https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html

    # 5. t-test independent, between classes
    # scipy.stats.ttest_ind, https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.stats.ttest_ind.html

    # 6. t-test related, between classes
    # scipy.stats.ttest_rel

    # 7. remove collinearity of feature vectors within class set (leave one)

    # 8. remove collinearity of feature vectors between classes (remove both)

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

def get_top_genes(MODELS=None, n_max = 1000, RexR=None):
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
    for mod in MODELS:
        if(mod['method'] in ['RandomForest', 'GBM', 'AdaBoost', 'ExtraTrees']): # RF, ET, GBM, ADA
            try:
                top_genomes_weights[mod['method']]=mod['model'].feature_importances_
                # column normalise
                top_genomes_weights[mod['method']] = top_genomes_weights[mod['method']]/top_genomes_weights[mod['method']].max()
            except AttributeError as e:
                print("model {} does not contain feature importances: {}".format(mod['method'] , e))
    # if multiple methods are used, only keep overlapping genes

    if(RexR.SET_NAME == 'MELA'):
        top_genomes_weights.index = RexR.DATA_merged_processed.drop(['target', 'ID'], axis=1).columns
    elif(RexR.SET_NAME == 'ALL_10'):
        drop_columns = RexR.DATA_merged_processed.columns[:21]
        top_genomes_weights.index = RexR.DATA_merged_processed.drop(drop_columns, axis=1).columns
        

    top_genomes_weights['MEAN'] = top_genomes_weights.mean(axis=1)
    top_genomes_weights = top_genomes_weights.sort_values(by='MEAN', ascending=False)[:n_max]
           
    ### Coefficients
    top_genomes_coeffs = pd.DataFrame()
    for mod in MODELS:
        if(mod['method'] in ['LR', 'SVM']):
            top_genomes_coeffs[mod['method']] = mod['model'].coef_[0,:]
            top_genomes_coeffs[mod['method']] = top_genomes_coeffs[mod['method']]/top_genomes_coeffs[mod['method']].max() #\
                                                                   #  -top_genomes[mod['method']].min())
                                                                     #+numpy.abs(top_genomes[mod['method']].min())
    if(RexR.SET_NAME == 'MELA'):            
        top_genomes_coeffs.index = RexR.DATA_merged_processed.drop(['target', 'ID'], axis=1).columns
    elif(RexR.SET_NAME == 'ALL_10'):
        drop_columns = RexR.DATA_merged_processed.columns[:21]
        top_genomes_coeffs.index = RexR.DATA_merged_processed.drop(drop_columns, axis=1).columns
    #top_genomes['ALL'] = top_genomes.sum(axis=1)
    top_genomes_coeffs['MEAN'] = top_genomes_coeffs.mean(axis=1)
    top_genomes_coeffs = top_genomes_coeffs.sort_values(by='MEAN', ascending=False)[:]    

    return top_genomes_weights, top_genomes_coeffs[-int(n_max/2):].append(top_genomes_coeffs[:int(n_max/2)]).sort_values(by="MEAN")

    ########
    ## couple genomes to probesets

    
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