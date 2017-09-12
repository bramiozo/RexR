from sklearn import preprocessing, svm, tree, ensemble, naive_bayes 
from sklearn import linear_model, neural_network, model_selection, metrics
from sklearn import discriminant_analysis, gaussian_process
import itertools
from sklearn.metrics import roc_curve, auc, roc_auc_score, log_loss, accuracy_score, confusion_matrix

#import xgboost as xgb
#import xgboost
import numpy as np
import _helpers
import copy
#import sys
#sys.setrecursionlimit(10000) 

def plot_cm(ax, y_true, y_pred, classes, title, th=0.5, cmap=plt.cm.Blues):
    y_pred_labels = (y_pred>th).astype(int)
    
    cm = confusion_matrix(y_true, y_pred_labels)
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

def plot_auc(ax, y_train, y_train_pred, y_test, y_test_pred, th=0.5):

    y_train_pred_labels = (y_train_pred>th).astype(int)
    y_test_pred_labels  = (y_test_pred>th).astype(int)

    fpr_train, tpr_train, _ = roc_curve(y_train,y_train_pred)
    roc_auc_train = auc(fpr_train, tpr_train)
    acc_train = accuracy_score(y_train, y_train_pred_labels)

    fpr_test, tpr_test, _ = roc_curve(y_test,y_test_pred)
    roc_auc_test = auc(fpr_test, tpr_test)
    acc_test = accuracy_score(y_test, y_test_pred_labels)

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


def classify_treatment(self, model_type='CART', 
                            features = 'genome', 
                            grouping= None, 
                            n_splits = 10,
                            re_normalise = False,
                            parameters = {},
                            pipeline = {"scaler": {"type": "minmax"},
                                        "dim_reduction": {"type": "PCA", "n_comp": 1000},
                                        "feature_selection": {"type":, "RFECV", "top_n": 100}}):  
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

    ######## PIPELINE ########
    ##########################
    ##########################
    df = self.DATA_merged
    if(self.DATA_merged_processed is None or re_normalise == False):
        print("+ "*30, 'Prepping data, this may take a while..')
        df = _helpers._preprocess(df, scaler = pipeline['scaler']['type'])
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
    elif(model_type == 'DNN'): # version 1: Keras
        print("NOT AVAILABLE YET")
    elif(model_type == 'CNN'): # version 1: Keras
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
    splitter = model_selection.StratifiedKFold(n_splits, random_state = self.SEED)
    print("+"*30,' RESULTS FOR CLASSIFICATION WITH GENOMIC DATA',"+"*30)
    preds = []
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
    
    '''
        model = rvm.rvm(x, y, noise = 0.01)
        preds = np.dot(x_test, model.wInferred);
        pred_ = np.append(preds, 1-preds[:], 1)
    '''
    if(model_type not in ['RVM', 'DNN', 'CNN', 'XGB', 'XGBoost'])
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
    
    preds = model.predict_proba(x_pred) # only for sklearn (compatible methods)

    print("+"*50)
    ################################################################
    ##### ADD PATIENT INFO TO PREDICTOR
    if(features == 'all'):
        print("+"*30,' RESULTS FOR CLASSIFICATION INCLUDING PATIENT DATA',"+"*30)
        p_x,y = _helpers._get_matrix(df, features = 'patient', target = 'Treatment risk group in ALL10')
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



    