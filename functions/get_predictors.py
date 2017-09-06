from sklearn import preprocessing, svm, tree, ensemble, naive_bayes 
from sklearn import linear_model, neural_network, model_selection, metrics
from sklearn import discriminant_analysis, gaussian_process
#import xgboost
import numpy as np
import _helpers
import copy
#import sys
#sys.setrecursionlimit(10000) 

def classify_treatment(self, model_type='CART', 
                            features = 'genome', 
                            grouping= None, 
                            n_splits = 10,
                            reduction = None,
                            n_comp = 55000,
                            re_normalise = False):  
    # model=['SVM', 'CART', 'LR', 'RandomForest', 'AdaBoost']
    # grouping=['first', 'median', 'mean']
    # features=['all', 'genomes']
    # reduction = ['PCA']
    # topN = None [None, N] 
    #
    #
    # streamlining code: create data pipelines http://sebastianraschka.com/Articles/2014_ensemble_classifier.html
    ############################################
    df = self.DATA_merged
    if(self.DATA_merged_processed is None or re_normalise == False):
        print("+ "*30, 'Prepping data, this may take a while..')
        df = _helpers._preprocess(df)
        print("- "*30, 'Grouping probesets')
        df = _helpers._group_patients(df, method = grouping)
        self.DATA_merged_processed = df
    else:
        df= self.DATA_merged_processed
    print("+ "*30, 'Creating X,y')
    x,y =_helpers._get_matrix(df, features = 'genomic', target = 'Treatment risk group in ALL10')      
    if reduction is not None:
        print("- "*30, 'Reducing dimensionality')
    if(reduction == 'PCA'):
            x, Reducer = _helpers.get_pca_transform(x, n_comp)
    elif(reduction == 'LDA'):
            x, Reducer = _helpers.get_lda_transform(x, y, n_comp)
    elif(reduction == 'RBM'):
            x, Reducer = _helpers.get_rbm_transform(x, y, n_comp)
    elif(reduction == 'genome_variance'):
            x, Reducer = _helpers.get_filtered_genomes(x, filter_type = None)
    #elif(reduction == 'auto_encoder'):
    #        x, Reducer = _helpers.get_autoencoded_genomes(x)
    ############################################
    models = []
    if(model_type == 'SVM'):
        model = svm.SVC(degree = 3, tol = 0.0001, C= 0.9, probability= True)
        models.append(('SVM', model))
    elif(model_type == 'CART'):
        model = tree.DecisionTreeClassifier(criterion='gini', splitter='best', 
                                max_depth=10, min_samples_split=2, min_samples_leaf=3, min_weight_fraction_leaf=0.0, 
                                max_features=None, random_state=self.SEED, max_leaf_nodes=None, min_impurity_split=1e-07, 
                                class_weight=None, presort=False)
        models.append(('CART', model))
    elif(model_type == 'LR'):
        model = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.9)
        models.append(('LR', model))
    elif(model_type == 'RandomForest'):
        model = ensemble.RandomForestClassifier(n_estimators=100, max_depth=25, n_jobs=-1, min_samples_split=5,\
                min_samples_leaf=5)
        models.append(('RF', model))
    elif(model_type == 'ExtraTrees'):
        model = ensemble.ExtraTreesClassifier(n_estimators=100, max_depth=75, n_jobs=-1, min_samples_split=5,\
                min_samples_leaf=5)
        models.append(('ET', model))
    elif(model_type == 'GBM'):
        model = ensemble.GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, 
            subsample=1.0, criterion='friedman_mse', min_samples_split=5, min_samples_leaf=5, 
            min_weight_fraction_leaf=0.0, max_depth=4, min_impurity_split=1e-07, init=None, 
            random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
        models.append(('GBM', model))
    elif(model_type == 'AdaBoost'):
        model = ensemble.AdaBoostClassifier(base_estimator=None, n_estimators=150, learning_rate=1.0, algorithm='SAMME.R', random_state=self.SEED)
        models.append(('ADA', model))
    elif(model_type == 'XGBoost'):
        print("NOT AVAILABLE YET")
    elif(model_type == 'RVM'):
        #import rvm
        #model = rvm(x, y, noise = 0.01)
        print("NOT AVAILABLE YET")
    elif(model_type == 'QDA'):
        model = discriminant_analysis.QuadraticDiscriminantAnalysis(priors = None, reg_param = 0.0)
        models.append(('QDA', model))
    elif(model_type == 'LDA'):
        model = discriminant_analysis.LinearDiscriminantAnalysis(priors = None, n_components = n_comp)
        models.append(('LDA', model))
    elif(model_type == 'GPC'):
        Kernel = 1.0 * gaussian_process.kernels.RBF(length_scale=1.0)
        model = gaussian_process.GaussianProcessClassifier(kernel=Kernel, 
                                                optimizer='fmin_l_bfgs_b', 
                                                n_restarts_optimizer=0, 
                                                max_iter_predict=100, 
                                                warm_start=False, 
                                                copy_X_train=True, 
                                                random_state=None, 
                                                multi_class='one_vs_rest', 
                                                n_jobs=1)
        models.append(('GPC', model))
    elif(model_type == 'NaiveBayes'):
        model = naive_bayes.GaussianNB()
        models.append(('GNB', model))
    elif(model_type == 'MLNN'):
        model = neural_network.MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                                       beta_1=0.9, beta_2=0.999, early_stopping=False,
                                       epsilon=1e-08, hidden_layer_sizes=(15, 7, 2), learning_rate='constant',
                                       learning_rate_init=0.001, max_iter=200, momentum=0.9,
                                       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                                       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
                                       warm_start=False)
        models.append(('MLNN', model))
    elif(model_type == 'ensemble'):
        models_ = [
            ("GNB", naive_bayes.GaussianNB()),
            ("SVM", svm.SVC(degree = 3, tol = 0.0001, C= 0.9, probability= True)),
            ("LogisticRegression", linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.9)),
            ("RandomForest", ensemble.RandomForestClassifier(n_estimators=100, max_depth=25, n_jobs=2, min_samples_split=5,\
                min_samples_leaf=5)),
            ("ExtraTrees", ensemble.ExtraTreesClassifier(n_estimators=100, max_depth=50, n_jobs=2, min_samples_split=5,\
                min_samples_leaf=5)),
            ("GBM", ensemble.GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, 
            subsample=1.0, criterion='friedman_mse', min_samples_split=5, min_samples_leaf=5, 
            min_weight_fraction_leaf=0.0, max_depth=4, min_impurity_split=1e-07, init=None, 
            random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),
            ("ADA", ensemble.AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)),
            ("CART", tree.DecisionTreeClassifier(criterion='gini', splitter='best', 
                                max_depth=10, min_samples_split=2, min_samples_leaf=3, min_weight_fraction_leaf=0.0, 
                                max_features=None, random_state=self.SEED, max_leaf_nodes=None, min_impurity_split=1e-07, 
                                class_weight=None, presort=False))
            ]
        models = copy.copy(models_)
        model = ensemble.VotingClassifier(models_, n_jobs = 2, voting = 'soft')
        models.append(("Ensembled", model))

    ############################################
    ############################## MODEL FITTING
    splitter = model_selection.StratifiedKFold(n_splits, random_state = self.SEED)
    print("+"*30,' RESULTS FOR CLASSIFICATION WITH GENOMIC DATA',"+"*30)
    preds = []
    for clf in models:
        pred, acc = _helpers._benchmark_classifier(clf, x, y, splitter, self.SEED)
        report = metrics.classification_report(y,pred)
        #acc = metrics.accuracy_score(y,pred)
        print('MODEL:', clf[0], 'accuracy: ',np.mean(acc), '+/-:', np.var(acc))
        print("+"*30,' Report', "+"*30)
        print(report)
        preds.append(pred)

    model.fit(x, y)
    var_columns = df.columns[21:]   
    x_pred = df.loc[:,var_columns].values  
    # apply dimensionality reduction
    #
    if(reduction == 'PCA'):
            x_pred = Reducer.transform(x_pred)
    elif(reduction == 'LDA'):
            x_pred = Reducer.transform(x_pred)
    elif(reduction == 'genome_variance'):
            x_pred = Reducer(x_pred)
    preds = model.predict(x_pred) 

    print("+"*50)
    ################################################################
    ##### ADD PATIENT INFO TO PREDICTOR
    if(features == 'all'):
        print("+"*30,' RESULTS FOR CLASSIFICATION INCLUDING PATIENT DATA',"+"*30)
        p_x,y = _helpers._get_matrix(df, features = 'patient', target = 'Treatment risk group in ALL10')
        pred = np.reshape(pred, (pred.shape[0], 1))
        print("---------")
        x = np.hstack([pred, p_x])
        for clf in models:
            pred, acc = _helpers._benchmark_classifier(clf,x,y,splitter, self.SEED)
            report = metrics.classification_report(y,pred)
            #acc = metrics.accuracy_score(y,pred)
            print('MODEL:', clf[0], 'accuracy: ',np.mean(acc), '+/-:', np.var(acc))            
            print("+"*30,' Report', "+"*30)
            print(report)   

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



    