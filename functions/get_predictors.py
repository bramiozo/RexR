from sklearn import preprocessing, svm, tree, ensemble, naive_bayes, neural_network, model_selection, metrics
import numpy as np
import _helpers
  
def classify_treatment_model(self, model='CART', 
                            features = 'genome', 
                            grouping= 'first', 
                            n_splits = 10,
                            reduction = None):  
    # model=['SVM', 'CART', 'LR', 'RandomForest', 'AdaBoost']
    # grouping=['first', 'median', 'mean']
    # features=['all', 'patient', genomes']
    # reduction = ['LDA', 'QDA', 'PCA', 'No']
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
    if(model == 'SVM'):
        model = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0)
    elif(model == 'CART'):
        model = tree.DecisionTreeClassifier(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0)
    elif(model == 'LR'):
        model = linear_model.LogisticRegression(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0)
    elif(model == 'RandomForest'):
        model = ensemble.RandomForestClassifier(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0)
    elif(model == 'ExtraTree'):
        model = ensemble.ExtraTreesClassifier(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0)
    elif(model == 'GBM'):
        model = ensemble.GradientBoostingClassifier(penalty='l2', tol=0.0001, C=1.0)
    elif(model == 'NB'):
        model = naive_bayes.GaussianNB(penalty='l2', loss='squared_hinge', 
                                        n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0)
    elif(model == 'MLNN'):
        model = neural_network.MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                                       beta_1=0.9, beta_2=0.999, early_stopping=False,
                                       epsilon=1e-08, hidden_layer_sizes=(100, 40, 10, 2), learning_rate='constant',
                                       learning_rate_init=0.001, max_iter=200, momentum=0.9,
                                       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                                       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
                                       warm_start=False)
    ############################################
    splitter = model_selection.StratifiedKFold(n_splits)
    print("+"*30,' RESULTS FOR CLASSIFICATION WITH GENOMIC DATA',"+"*30)
    for train_index, test_index in splitter.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index] 
        model.fit(x_train,y_train)
        pred = model.predict(x_test)
        print(metrics.accuracy_score(y_test,pred))

    model.fit(x, y)
    pred = model.predict(x)
    print("+"*50)
    ################################################################
    if(features == 'all'):
        print("+"*30,' RESULTS FOR CLASSIFICATION INCLUDING PATIENT DATA',"+"*30)
        x,y = _helpers._get_matrix(df, type = 'patient', target = 'Treatment risk group in ALL10')
        x = x.append(pred, axis=1)
        for train_index, test_index in splitter.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index] 
            #print(train_index.shape, test_index.shape)
            model.fit(x_train,y_train)
            pred = model.predict(x_test)
            print(metrics.accuracy_score(y_test,pred))        

        model.fit(x, y)    

    return model

def get_top_genes(model, n = 100):
    print("---", model._coef[:n])



    return True


def estimate_survival(output_type = 'years'): # output_type=['years', '']
    # Regression trees
    # Gaussian processes
    # GradientBoostedRegressor



    return survival



    