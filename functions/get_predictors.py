from sklearn import preprocessing, svm, tree, ensemble, naive_bayes 
from sklearn import linear_model, neural_network, model_selection, metrics
from sklearn import discriminant_analysis, gaussian_process
from xgboost.sklearn import XGBClassifier as xgb
import lightgbm as lgb
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import MaxPooling1D
from keras.callbacks import Callback
from keras.layers.convolutional import Conv1D
from keras.layers import Input 

import matplotlib.pyplot as plt
import _helpers
import copy
import itertools
#import sys
#sys.setrecursionlimit(10000) 

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
    if self.DATA_merged_processed is None and pipeline['scaler'] is not None:
        print("+ "*30, 'Prepping data, this may take a while..')
        df = _helpers._preprocess(df, scaler = pipeline['scaler']['type'], Rclass = self)
        print("- "*30, 'Grouping probesets')
        df = _helpers._group_patients(df, method = pipeline['pre_processing']['patient_grouping'], Rclass = self)
        if pipeline['pre_processing']['bias_removal'] == True:
            print("- "*30, 'Removing cohort biases')
            df = _helpers._cohort_correction(df)
        self.DATA_merged_processed = df
    else:
        df= self.DATA_merged_processed
    print("+ "*30, 'Creating X,y')
    if(self.X_GENOME is None):
        x,y =_helpers._get_matrix(df, features = 'genomic', target = parameters['target'], Rclass = self)
        self.X_GENOME = x
        self.Y_CLASS = y
    else:
        x = self.X_GENOME
        y = self.Y_CLASS    

    if pipeline['feature_selection']['type'] is not None:        
        NOW_HASH = hash(pipeline['feature_selection']['type']+
                        pipeline['feature_selection']['method']+
                        str(pipeline['feature_selection']['pvalue'])+
                        str(x.trace()))

        if NOW_HASH == self.PREP_HASH:
            Selector = self.PREP_SELECTOR
        elif self.PREP_HASH is None:
            print("- "*30, 'Selecting features using a {} filter'.format(pipeline['feature_selection']['type']))
            x_ = np.copy(x)
            or_cols = x_.shape[1]
            if pipeline['feature_selection']['type'] == 'low_variance':
                x, Selector = _helpers.get_filtered_genomes(x_, y, alpha=pipeline['feature_selection']['pvalue'],
                                                     filter_type = pipeline['feature_selection']['method'])

            print("- "*30, 'Kept {} of {} features using {} with p = {}'.format(str(x.shape[1]),
                                                      str(or_cols),
                                                      pipeline['feature_selection']['method'],
                                                      pipeline['feature_selection']['pvalue']))
            
            self.X_GENOME = x
            self.PREP_HASH = hash(pipeline['feature_selection']['type']+
                                 pipeline['feature_selection']['method']+
                                 str(pipeline['feature_selection']['pvalue'])+
                                 str(x.trace()))
            self.PREP_SELECTOR = Selector
            if(pipeline['feature_selection']['method']=='mannwhitney'):
                self.PREP_DEL = Selector.transform(x)[1]
            elif(pipeline['feature_selection']['method']=='FDR'):
                self.PREP_DEL = [idx for idx, item in enumerate(Selector.get_support()) if item == False]

    if pipeline['dim_reduction']['type'] is not None:
        print("- "*30, 'Reducing dimensionality using {}'.format(pipeline['dim_reduction']['type']))
        x_ = np.copy(x)
        x, Reducer = _helpers.get_dim_reduction(x_, y, n_comp = pipeline['dim_reduction']['n_comp'], 
                                                method = pipeline['dim_reduction']['type'], Rclass = self)


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
    elif(model_type == 'LGBM'):
        pars = parameters['LGBM'] 
        model = lgb.LGBMClassifier(**pars)      
        models.append(('LGBM', model))
    elif(model_type in ['AdaBoost','Ada']):
        pars = parameters['ADA']
        model = ensemble.AdaBoostClassifier(**pars)
        models.append(('ADA', model))
    elif(model_type in ['XGBoost', 'XGB']):
        pars = parameters['XGB']
        model = xgb(**pars)
        models.append(('XGB', model))
    elif(model_type == 'DNN'): # version 1: Keras, not very useful atm given that we have so few samples.
        model = Sequential()
        input_dim = x.shape[1]
        model.add(Dense(356, input_shape=(input_dim,), activation='tanh'))
        model.add(Dense(356, input_shape=(input_dim,), activation='tanh'))
        model.add(Dense(256, activation='relu')) # relu, selu, tanh, sigmoid
        model.add(Dense(256, activation='relu')) # relu, selu, tanh, sigmoid
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1,  activation='sigmoid'))   
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])   
        models.append(('DNN', model))
    elif(model_type == 'CNN'): # version 1: Keras, load common cnn architecture like Inception       
        x = np.expand_dims(x, axis=2)
        arch = parameters['CNN']['architecture']
        input_dim = x.shape[1]
        if (arch in ['vgg16', 'vgg19', 'resnet50', 'inception', 'xception']):
            # load directly using keras
            # these models assume 2D-images, hence the data has to be re-shaped.
            # simply re-shaping, or re-shaping according to the Hilbert curve
            # how if input_dim is a prime number?? :-D
            if arch == 'resnet50':
                from keras.applications import ResNet50
                model = ResNet50(weights=None, input_tensor=Input(shape=(input_dim,1)))            
            elif arch == 'vgg16':
                from keras.applications import VGG16
                model = VGG16(weights=None, input_tensor=Input(shape=(input_dim,1)))
            elif arch == 'vgg19':
                from keras.applications import VGG19
                model = VGG19(weights=None, input_tensor=Input(shape=(input_dim,1))) 
            elif arch == 'inception':
                from keras.applications import InceptionV3
                model = InceptionV3(weights=None, input_tensor=Input(shape=(input_dim,1)))  
            elif arch == 'xception':
                from keras.applications import Xception
                model = Xception(weights=None, input_tensor=Input(shape=(input_dim,1)))
            model.compile(optimizer='rmsprop', loss='binary_crossentropy')    
            models.append(('CNN', model))
        else:
            # read h5 from model_location or use custom model
            model = Sequential()
            model.add(Conv1D(32, 6, input_shape=(input_dim, 1)))
            model.add(Activation('relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Conv1D(32, 6))
            model.add(Activation('relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Conv1D(64, 6))
            model.add(Activation('relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
            model.add(Dense(64))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1))
            model.add(Activation('sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
            models.append(('CNN', model))
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
            ("SVM", svm.SVC(**parameters['SVM'])),
            ("LogisticRegression", linear_model.LogisticRegression(**parameters['LR'])),
            ("RandomForest", ensemble.RandomForestClassifier(**parameters['RF'])),
            ("ExtraTrees", ensemble.ExtraTreesClassifier(**parameters['ET'])),
            ("GBM", ensemble.GradientBoostingClassifier(**parameters['GBM'])),
            ("ADA", ensemble.AdaBoostClassifier(**parameters['ADA'])),
            ("CART", tree.DecisionTreeClassifier(**parameters['CART'])),
            ("XGB", xgb(**parameters['XGB']))
            ]
        models = copy.copy(models_)
        model = ensemble.VotingClassifier(models_, n_jobs = -1 , voting = 'soft')
        models.append(("Ensembled", model))

    ############################################
    ############################## MODEL FITTING
    splitter = model_selection.StratifiedKFold(parameters['n_splits'], random_state = self.SEED)
    print("+"*30,' RESULTS FOR CLASSIFICATION WITH GENOMIC DATA',"+"*30)
    print("+"*20, "..processing feature array {} and class vector {}".format(np.shape(x), np.shape(y)))
    preds = []
    accuracy = []
    for clf in models:
        if clf[0] == 'RVM':
            pred, acc = _helpers._benchmark_classifier(clf, x, y, splitter, self.SEED, framework = 'custom_rvm', Rclass = self)
        elif clf[0] in ['DNN', 'CNN']:
            pred, acc = _helpers._benchmark_classifier(clf, x, y, splitter, self.SEED, framework = 'keras', Rclass = self)
        else:
            pred, acc = _helpers._benchmark_classifier(clf, x, y, splitter, self.SEED, framework = 'sklearn', Rclass = self)

        #acc = metrics.accuracy_score(y,pred)
        accuracy.append({'model': clf[0], 'acc': np.mean(acc), 'var' : np.var(acc)})
        print('MODEL:', clf[0], 'accuracy: ',np.mean(acc), '+/-:', np.var(acc))
        if self.VIZ == True:
            report = metrics.classification_report(y,pred)
            print("+"*30,' Report', "+"*30)
            print(report)
    
    '''
        model = rvm.rvm(x, y, noise = 0.01)
        preds = np.dot(x_test, model.wInferred);
        pred_ = np.append(preds, 1-preds[:], 1)
    '''
    if(model_type not in ['RVM', 'DNN', 'CNN']): # 'XGB', 'XGBoost'
        model.fit(x, y) 
    elif(model_type in ['DNN', 'CNN']): 
        model.fit(x, y, batch_size = 10, epochs = 5, verbose = 0, callbacks=[BL]) 
    elif(model_type == 'RVM'):
        model = rvm.rvm(x, y, noise = 0.01)
        model.iterateUntilConvergence()
    elif(model_type.lower() in ['lgbm', 'lightgbm']):
        x_ = lgb.Dataset(x, label=None, max_bin=8192) #
        ''' params = {'early_stopping_rounds':10,  
                   'eval_metric':'logloss', 
                   'train_set': d_train, 
                   'num_boost_round':7000, 
                   'verbose_eval':1000,
                   'verbose'=True}'''       

        model.fit(x_, y, early_stopping_rounds=10, verbose=True)

    if self.SET_NAME == 'ALL_10':
        var_columns = df.columns[21:]   
    elif self.SET_NAME == 'MELA':
        var_columns = df.loc[:, (df.columns!=parameters['target']) &  (df.columns!=parameters['ID'])].columns
    x_pred = df.loc[:,var_columns].values  

    # apply feature selection
    #
    if(pipeline['feature_selection']['type'] is not None):
        if pipeline['feature_selection']['method'] == 'mannwhitney':
            x_pred = Selector.transform(x_pred)[0]
        else:
            x_pred = Selector.transform(x_pred)

    # apply dimensionality reduction
    #
    if(pipeline['dim_reduction']['type'] == 'PCA'):
        x_pred = Reducer.transform(x_pred)
    elif(pipeline['dim_reduction']['type']  == 'LDA'):
        x_pred = Reducer.transform(x_pred)
    elif(pipeline['dim_reduction']['type'] == 'PLS'):
        x_pred = Reducer.transform(x_pred)
    elif(pipeline['dim_reduction']['type']  == 'genome_variance'):
        x_pred = Reducer(x_pred)

    
    if(model_type not in ['RVM', 'DNN', 'CNN']): # , 'XGB', 'XGBoost'
        preds = model.predict_proba(x_pred) # only for sklearn (compatible methods)
    #elif model_type == 'DNN':
    #    #preds = 
    elif model_type == 'RVM':
        preds   = np.reshape(np.dot(x_pred, model.wInferred), newshape=[len(x_pred),])/2+0.5    
    elif model_type == 'DNN':
        preds   = model.predict_on_batch(np.array(x_pred))[:,0]  
    elif model_type == 'CNN':
        preds   = model.predict(np.expand_dims(x_pred, axis=2))[:,0] 
    elif model_type.lower() in ['lgbm', 'lightgbm']:
        preds = model.predict_proba(X_test, num_iteration = model.best_iteration)

    ################
    ################

    print("+"*50)
    ################################################################
    ##### ADD PATIENT INFO TO PREDICTOR, is specific to ALL-10
    if(features == 'all'):
        #### This assumes that the previous predictions are suitable as features.
        ##################################
        print("+"*30,' RESULTS FOR CLASSIFICATION INCLUDING PATIENT DATA',"+"*30)
        p_x,y = _helpers._get_matrix(df, features = 'patient', target = parameters['target'], Rclass = self)
        scaler = preprocessing.StandardScaler()
        p_x = scaler.fit_transform(p_x)
        pred = np.reshape(pred, (pred.shape[0], 1))
        print("---------")
        x = np.hstack([pred, p_x])
        for clf in models:
            if clf[0] == 'RVM':
                pred, acc = _helpers._benchmark_classifier(clf, x, y, splitter, self.SEED, 
                                                        framework = 'custom_rvm', Rclass=self)
            else:
                pred, acc = _helpers._benchmark_classifier(clf, x, y, splitter, self.SEED, 
                                                            framework = 'sklearn', Rclass=self)            
            report = metrics.classification_report(y,pred)
            #acc = metrics.accuracy_score(y,pred)
            print('MODEL:', clf[0], 'accuracy: ',np.mean(acc), '+/-:', np.var(acc))            
            if self.VIZ == True:
                print("+"*30,' Report', "+"*30)
                print(report)   

        if(model_type not in ['RVM', 'DNN', 'CNN']): # , 'XGB', 'XGBoost'
            model.fit(x, y)
        # total x
        var_columns = ["Age", "WhiteBloodCellcount", "Gender"]
        df[var_columns] = df[var_columns].fillna(0.0)       
        x = df.loc[:,var_columns].values
        preds = np.reshape(preds, (preds.shape[0], 1))
        x = np.hstack([preds, x])
        preds = model.predict_proba(x)
    

    return preds, model, accuracy


    # hyper optimalisation routines.
    ## start with grid search



    # relapse predictor / survival rate



    