from sklearn import preprocessing, svm, tree, ensemble, naive_bayes, linear_model, neural_network, model_selection, metrics
import numpy as np
import _helpers
import copy
#import sys
#sys.setrecursionlimit(10000) 

def classify_treatment(self, model='CART', 
                            features = 'genome', 
                            grouping= 'first', 
                            n_splits = 10,
                            reduction = None):  
    # model=['SVM', 'CART', 'LR', 'RandomForest', 'AdaBoost']
    # grouping=['first', 'median', 'mean']
    # features=['all', 'patient', genomes']
    # reduction = ['LDA', 'QDA', 'PCA', 'No']
    # topN = None [None, N] 
    ############################################
    df = self.DATA_merged
    df = _helpers._preprocess(df)
    df = _helpers._group_patients(df, method = grouping)
    x,y =_helpers._get_matrix(df, type = 'genomic', target = 'Treatment risk group in ALL10')      
    if(reduction == 'QDA'):
            x,y = self.get_dimension_reduction.get_principal_components(x, y, n_comp)
    elif(reduction == 'LDA'):
            x,y = self.get_dimension_reduction.get_linear_discriminant_analysis(x, y)
    elif(reduction == 'PCA'):
            x,y = self.get_dimension_reduction.get_quadrant_discriminant_analysis(x, y)
    ############################################
    models = []
    if(model == 'SVM'):
        model = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0)
        models.append(('SVM', model))
    elif(model == 'CART'):
        model = tree.DecisionTreeClassifier(criterion='gini', splitter='best', 
                                max_depth=10, min_samples_split=2, min_samples_leaf=3, min_weight_fraction_leaf=0.0, 
                                max_features=None, random_state=self.SEED, max_leaf_nodes=None, min_impurity_split=1e-07, 
                                class_weight=None, presort=False)
        models.append(('CART', model))
    elif(model == 'LogisticRegression'):
        model = linear_model.LogisticRegression(penalty='l2', loss='squared_hinge', dual=False, tol=0.0001, C=0.9)
        models.append(('LR', model))
    elif(model == 'RandomForest'):
        model = ensemble.RandomForestClassifier(n_estimators=100, max_depth=25, n_jobs=-1, min_samples_split=5,\
                min_samples_leaf=5)
        models.append(('RF', model))
    elif(model == 'ExtraTrees'):
        model = ensemble.ExtraTreesClassifier(n_estimators=100, max_depth=75, n_jobs=-1, min_samples_split=5,\
                min_samples_leaf=5)
        models.append(('ET', model))
    elif(model == 'GBM'):
        model = ensemble.GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, 
            subsample=1.0, criterion='friedman_mse', min_samples_split=5, min_samples_leaf=5, 
            min_weight_fraction_leaf=0.0, max_depth=4, min_impurity_split=1e-07, init=None, 
            random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
        models.append(('GBM', model))
    elif(model == 'NaiveBayes'):
        model = naive_bayes.GaussianNB(penalty='l2', loss='squared_hinge', 
                                        n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0)
        models.append(('NB', model))
    elif(model == 'MLNN'):
        model = neural_network.MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                                       beta_1=0.9, beta_2=0.999, early_stopping=False,
                                       epsilon=1e-08, hidden_layer_sizes=(15, 7, 2), learning_rate='constant',
                                       learning_rate_init=0.001, max_iter=200, momentum=0.9,
                                       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                                       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
                                       warm_start=False)
        models.append(('MLNN', model))
    elif(model == 'ensemble'):
        models_ = [
            ("SVM", svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=0.9)),
            ("LogisticRegression", linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.9)),
            ("RandomForest", ensemble.RandomForestClassifier(n_estimators=100, max_depth=25, n_jobs=-1, min_samples_split=5,\
                min_samples_leaf=5)),
            ("ExtraTrees", ensemble.ExtraTreesClassifier(n_estimators=100, max_depth=100, n_jobs=-1, min_samples_split=5,\
                min_samples_leaf=5)),
            ("GBM", ensemble.GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, 
            subsample=1.0, criterion='friedman_mse', min_samples_split=5, min_samples_leaf=5, 
            min_weight_fraction_leaf=0.0, max_depth=4, min_impurity_split=1e-07, init=None, 
            random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),
            ("CART", tree.DecisionTreeClassifier(criterion='gini', splitter='best', 
                                max_depth=10, min_samples_split=2, min_samples_leaf=3, min_weight_fraction_leaf=0.0, 
                                max_features=None, random_state=self.SEED, max_leaf_nodes=None, min_impurity_split=1e-07, 
                                class_weight=None, presort=False))
            ]
        models = copy.copy(models_)
        model = ensemble.VotingClassifier(models_, n_jobs=2)
        models.append(("Ensembled", model))

    ############################################
    ############################## MODEL FITTING
    splitter = model_selection.StratifiedKFold(n_splits, random_state = self.SEED)
    print("+"*30,' RESULTS FOR CLASSIFICATION WITH GENOMIC DATA',"+"*30)
    preds = []
    for clf in models:
        pred = _helpers._benchmark_classifier(clf, x, y, splitter, self.SEED)
        report = metrics.classification_report(y,pred)
        acc = metrics.accuracy_score(y,pred)
        print('MODEL:', clf[0], acc)
        print("+"*30,' Report', "+"*30)
        print(report)
        preds.append(pred)

    model.fit(x, y)
    print("+"*50)
    ################################################################
    ##### ADD PATIENT INFO TO PREDICTOR
    if(features == 'all'):
        print("+"*30,' RESULTS FOR CLASSIFICATION INCLUDING PATIENT DATA',"+"*30)
        p_x,y = _helpers._get_matrix(df, type = 'patient', target = 'Treatment risk group in ALL10')
        pred = np.reshape(pred, (pred.shape[0], 1))
        print("---------")
        x = np.hstack([pred, p_x])
        for clf in models:
            pred = _helpers._benchmark_classifier(clf,x,y,splitter, self.SEED)
            report = metrics.classification_report(y,pred)
            acc = metrics.accuracy_score(y,pred)
            print(acc)
            print("+"*30,' Report', "+"*30)
            print(report)            
            preds.append(pred)     

        model.fit(x, y)    

    return preds, model


def get_top_genes(model, n = 10):
    coef = model.coef_
    coef = np.reshape(coef, (coef.shape[1],))
    ordering = np.argsort(np.abs(coef))
    topN = ordering[-n:]

    gene_columns = self.DATA_merged.columns[21:].values

    top_genes = []
    for test in topN:
        top_genes.append({'probeset': gene_columns[test], 'rank_coeff': test})

    #coef_list = []
    #for coef in model._coef:
    #    coef_list.append(coef)

    return top_genes



    