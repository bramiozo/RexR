from sklearn import preprocessing, svm, tree, ensemble, naive_bayes 
from sklearn import linear_model, neural_network, model_selection, metrics
from sklearn import discriminant_analysis, gaussian_process
import itertools

import matplotlib.pyplot as plt

#import xgboost as xgb
#import xgboost
import numpy as np
import _helpers
import copy
#import sys
#sys.setrecursionlimit(10000) 

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import Callback

import pandas as pd
import numpy as np

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

def classify_treatment(self, model_type='CART', 
                            features = 'genome', 
                            parameters = {},
                            pipeline = {}):  
    # model=['SVM', 'CART', 'LR', 'RandomForest', 'AdaBoost']
    # grouping=['first', 'median', 'mean']
    # features=['all', 'genomes']
    # reduction = ['PCA']
    # topN = None [None, N] 
    # parameters: dict of dicts, **kwargs for functions
    #
    # streamlining code: create data pipelines http://sebastianraschka.com/Articles/2014_ensemble_classifier.html
    ############################################
    if parameters == {}: # empty dict, so fetch defaults
        parameters = self.MODEL_PARAMETERS

    if pipeline == {}:    
        pipeline = self.PIPELINE_PARAMETERS
    ##########################
    ######## PIPELINE ########
    ##########################
    ##########################
    df = self.DATA_merged
    if(self.DATA_merged_processed is None):
        print("+ "*30, 'Prepping data, this may take a while..')
        df = _helpers._preprocess(df, scaler = pipeline['scaler']['type'])
        print("- "*30, 'Grouping probesets')
        df = _helpers._group_patients(df, method = pipeline['pre_processing']['patient_grouping'])
        if pipeline['pre_processing']['bias_removal'] == True:
            print("- "*30, 'Removing cohort biases')
            df = _helpers._cohort_correction(df)
        self.DATA_merged_processed = df
    else:
        df= self.DATA_merged_processed
    print("+ "*30, 'Creating X,y')
    x,y =_helpers._get_matrix(df, features = 'genomic', target = 'Treatment_risk_group_in_ALL10')      
    if pipeline['dim_reduction']['type'] is not None:
        print("- "*30, 'Reducing dimensionality')
        x_ = np.copy(x)
        if(pipeline['dim_reduction']['type'] == 'PCA'):
                x, Reducer = _helpers.get_pca_transform(x_, pipeline['dim_reduction']['n_comp'], self)      
        elif(pipeline['dim_reduction']['type'] == 'LDA'):
                x, Reducer = _helpers.get_lda_transform(x_, y, pipeline['dim_reduction']['n_comp'], self)
        elif(pipeline['dim_reduction']['type'] == 'RBM'):
                x, Reducer = _helpers.get_rbm_transform(x_, y, pipeline['dim_reduction']['n_comp'], self)
        elif(pipeline['dim_reduction']['type'] == 'genome_variance'):
                x, Reducer = _helpers.get_filtered_genomes(x_, filter_type = None)

    # if dimension reduction AND feature selection, then perform FeatureUnion
    ## http://scikit-learn.org/stable/auto_examples/plot_feature_stacker.html#sphx-glr-auto-examples-plot-feature-stacker-py
    
    #########################
    #########################
    models = []
    if(model_type == 'SVM'):
        pars = parameters['SVM']
        model = svm.SVC(**pars)
        models.append(('SVM', model))
    elif(model_type == 'CART'):
        pars = parameters['CART']
        model = tree.DecisionTreeClassifier(**pars)
        models.append(('CART', model))
    elif(model_type == 'LR'):
        pars  = parameters['LR']
        model = linear_model.LogisticRegression(**pars)
        models.append(('LR', model))
    elif(model_type in ['RandomForest', 'RF']):
        pars = parameters['RF']
        model = ensemble.RandomForestClassifier(**pars)
        models.append(('RF', model))
    elif(model_type == 'ExtraTrees'):
        pars = parameters['ET']
        model = ensemble.ExtraTreesClassifier(**pars)
        models.append(('ET', model))
    elif(model_type == 'GBM'):
        pars = parameters['GBM']
        model = ensemble.GradientBoostingClassifier(**pars)
        models.append(('GBM', model))
    elif(model_type == 'AdaBoost'):
        pars = parameters['ADA']
        model = ensemble.AdaBoostClassifier(**pars)
        models.append(('ADA', model))
    elif(model_type == 'XGBoost'):
        print("NOT AVAILABLE YET")
    elif(model_type == 'DNN'): # version 1: Keras, not very useful atm given that we have so few samples.
        model = Sequential()
        input_dim = x.shape[1]
        model.add(Dense(256, input_shape=(input_dim,), activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(10, activation='sigmoid'))
        model.add(Dense(1,  activation='sigmoid'))   
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])   
        models.append(('DNN', model))
    elif(model_type == 'CNN'): # version 1: Keras, load common cnn architecture like Inception
        print("NOT AVAILABLE YET")
    elif(model_type == 'RVM'):
        import rvm
        models.append(('RVM', None))
    elif(model_type == 'EBE'): # Extremely Biased Estimator
        print("NOT AVAILABLE YET")
    elif(model_type == 'QDA'):
        model = discriminant_analysis.QuadraticDiscriminantAnalysis(priors = None, reg_param = 0.0)
        models.append(('QDA', model))
    elif(model_type == 'LDA'):
        model = discriminant_analysis.LinearDiscriminantAnalysis(priors = None, n_components = n_comp)
        models.append(('LDA', model))
    elif(model_type == 'GPC'):
        Kernel = 1.0 * gaussian_process.kernels.RBF(length_scale=1.0)
        pars = parameters['GPC']
        model = gaussian_process.GaussianProcessClassifier(kernel=Kernel, **pars)
        models.append(('GPC', model))
    elif(model_type == 'NaiveBayes'):
        model = naive_bayes.GaussianNB()
        models.append(('GNB', model))
    elif(model_type == 'MLNN'):
        pars = parameters['MLNN']
        model = neural_network.MLPClassifier(**pars) #  solver = 'lbfgs'
        models.append(('MLNN', model))
    elif(model_type == 'ensemble'):
        models_ = [
            ("GNB", naive_bayes.GaussianNB()),
            ("SVM", svm.SVC(**parameters['SVM'])),
            ("LogisticRegression", linear_model.LogisticRegression(**parameters['LR'])),
            ("RandomForest", ensemble.RandomForestClassifier(**parameters['RF'])),
            ("ExtraTrees", ensemble.ExtraTreesClassifier(**parameters['ET'])),
            ("GBM", ensemble.GradientBoostingClassifier(**parameters['GBM'])),
            ("ADA", ensemble.AdaBoostClassifier(**parameters['ADA'])),
            ("CART", tree.DecisionTreeClassifier(**parameters['CART']))
            ]
        models = copy.copy(models_)
        model = ensemble.VotingClassifier(models_, n_jobs = -1 , voting = 'soft')
        models.append(("Ensembled", model))

    ############################################
    ############################## MODEL FITTING
    splitter = model_selection.StratifiedKFold(parameters['n_splits'], random_state = self.SEED)
    print("+"*30,' RESULTS FOR CLASSIFICATION WITH GENOMIC DATA',"+"*30)
    preds = []
    for clf in models:
        if clf[0] == 'RVM':
            pred, acc = _helpers._benchmark_classifier(clf, x, y, splitter, self.SEED, framework = 'custom_rvm')
        elif clf[0] == 'DNN':
            pred, acc = _helpers._benchmark_classifier(clf, x, y, splitter, self.SEED, framework = 'keras')
        else:
            pred, acc = _helpers._benchmark_classifier(clf, x, y, splitter, self.SEED, framework = 'sklearn')
        report = metrics.classification_report(y,pred)
        #acc = metrics.accuracy_score(y,pred)
        print('MODEL:', clf[0], 'accuracy: ',np.mean(acc), '+/-:', np.var(acc))
        print("+"*30,' Report', "+"*30)
        print(report)
    
    '''
        model = rvm.rvm(x, y, noise = 0.01)
        preds = np.dot(x_test, model.wInferred);
        pred_ = np.append(preds, 1-preds[:], 1)
    '''
    if(model_type not in ['RVM', 'DNN', 'CNN', 'XGB', 'XGBoost']):
        model.fit(x, y) 
    elif(model_type == 'DNN'): 
        model.fit(x, y, batch_size = 10, epochs = 100, verbose = 0, callbacks=[BL]) 
        #   
    elif(model_type == 'RVM'):
        model = rvm.rvm(x, y, noise = 0.01)
        model.iterateUntilConvergence()

    var_columns = df.columns[21:]   
    x_pred = df.loc[:,var_columns].values  
    # apply dimensionality reduction
    #
    if(pipeline['dim_reduction']['type'] == 'PCA'):
        x_pred = Reducer.transform(x_pred)
    elif(pipeline['dim_reduction']['type']  == 'LDA'):
        x_pred = Reducer.transform(x_pred)
    elif(pipeline['dim_reduction']['type']  == 'genome_variance'):
        x_pred = Reducer(x_pred)
    
    if(model_type not in ['RVM', 'DNN', 'CNN', 'XGB', 'XGBoost']):
        preds = model.predict_proba(x_pred) # only for sklearn (compatible methods)
    #elif model_type == 'DNN':
    #    #preds = 
    elif model_type == 'RVM':
        preds   = np.reshape(np.dot(x_pred, model.wInferred), newshape=[len(x_pred),])/2+0.5    
    elif model_type == 'DNN':
        preds   = model.predict_on_batch(np.array(x_pred))[:,0]    

    ################
    ################

    print("+"*50)
    ################################################################
    ##### ADD PATIENT INFO TO PREDICTOR
    if(features == 'all'):
        #### This assumes that the previous predictions are suitable as features.
        ##################################
        print("+"*30,' RESULTS FOR CLASSIFICATION INCLUDING PATIENT DATA',"+"*30)
        p_x,y = _helpers._get_matrix(df, features = 'patient', target = 'Treatment_risk_group_in_ALL10')
        scaler = preprocessing.StandardScaler()
        p_x = scaler.fit_transform(p_x)
        pred = np.reshape(pred, (pred.shape[0], 1))
        print("---------")
        x = np.hstack([pred, p_x])
        for clf in models:
            if clf[0] == 'RVM':
                pred, acc = _helpers._benchmark_classifier(clf, x, y, splitter, self.SEED, framework = 'custom_rvm')
            else:
                pred, acc = _helpers._benchmark_classifier(clf, x, y, splitter, self.SEED, framework = 'sklearn')            
            report = metrics.classification_report(y,pred)
            #acc = metrics.accuracy_score(y,pred)
            print('MODEL:', clf[0], 'accuracy: ',np.mean(acc), '+/-:', np.var(acc))            
            print("+"*30,' Report', "+"*30)
            print(report)   

        if(model_type not in ['RVM', 'DNN', 'CNN', 'XGB', 'XGBoost']):
            model.fit(x, y)
        # total x
        var_columns = ["Age", "WhiteBloodCellcount", "Gender"]
        df[var_columns] = df[var_columns].fillna(0.0)       
        x = df.loc[:,var_columns].values
        preds = np.reshape(preds, (preds.shape[0], 1))
        x = np.hstack([preds, x])
        preds = model.predict_proba(x)
    

    return preds, model


    # hyper optimalisation routines.



    # relapse predictor / survival rate



    