import pandas as pd
from collections import Counter 
import numpy as np


####
# AUTHORS
# Bram van Es (NLeSC)
# Sebastiaan de Jong (ABN AMRO)
# Wybe Rozema (ABN AMRO)
# Sabrina Wandl (ABN AMRO)
# Nick Heuls (ABN AMRO)

# Jules Meijerink (Erasmus)
# Tjebbe Tauber (ABN AMRO)
#################

'''
Suggested features
--------------------------

Classification:
apply denoising
apply dimensionality reduction
apply probabilistic classifiers


Probeset drivers for treatment:


Probeset drivers for cancer types/pathways:


Survival estimation given a particular treatment:

to-do's (Q1 2018):
- [x] deep learner
- [x] LightGBM
- [ ] t-sne / optics analyser
- [x] ROC/confusion matrix visualiser
- [x] patient similarity
- [x] xgboost
- [ ] cohort-bias reducer
- [ ] conditional survival estimator
- [ ] GEO DataSets lib integration
- [ ] gene importance visualiser
'''




class RexR():
    # constants
    X_GENOME = None
    Y_CLASS = None
    DATA_all_samples = None
    DATA_patients = None
    DATA_Tnormal = None
    DATA_merged = None
    DATA_merged_processed = None
    DATA_loc = None
    MODEL_PARAMETERS = {}
    FEATURE_SELECTION_PARAMETERS = {}
    DIMENSION_REDUCTION_PARAMETERS = {}
    PIPELINE_PARAMETERS = {}
    READ_PARAMETERS = {}
    
    PREP_HASH = None
    PREP_SELECTOR = None
    PREP_DEL = None

    write_out = None
    SEED = 1234
    DEBUG = False
    VIZ = True # print all plot/accuracies to screen
    SET_NAME = None 

    def __init__(self, datalocation = None, seed = 2412, debug = False, write_out = False, set_name = 'ALL_10'):
        print("+"*30, 'Firing up RexR!', "+"*30)
        self.DATA_loc = datalocation
        self.SEED = seed
        self.DEBUG = debug
        self.SET_NAME = set_name
        self.write_out = write_out
        self.MODEL_PARAMETERS = {
            "target": 'Treatment_risk_group_in_ALL10',
            "ID": 'ID',
            "n_splits": 5,
            "SVM":{'kernel': 'linear', 'gamma': 'auto', 'tol': 0.0005, 'C':  0.9, 'probability' : True, 'max_iter': 3000}, # kernel linear, rbf, poly, sigmoid
            #"LSVM": {'C':1.0, 'class_weight':None, 'dual':True, 'fit_intercept':True,
            #            'intercept_scaling':1, 'loss':'squared_hinge', 'max_iter':1000,
            #            'multi_class':'ovr', 'penalty':'l2', 'random_state':0, 'tol': 0.0001, 'verbose':0}, 
            "RF": {'n_estimators': 200, 'max_depth': 10, 'n_jobs': -1, 'min_samples_split': 10, 'min_samples_leaf': 5},
            "MLNN": {'activation':'tanh', 'alpha':1e-04, 'batch_size': 10,
                    'beta_1':0.9, 'beta_2':0.999, 'early_stopping':False,
                    'epsilon':1e-06, 'hidden_layer_sizes':(60, 30, 15, 7, 2), 'learning_rate':'adaptive',
                    'learning_rate_init':0.001, 'max_iter':200, 'momentum':0.9,
                    'nesterovs_momentum':True, 'power_t':0.5, 'random_state':1, 'shuffle':True,
                    'solver':'adam', 'tol':0.001, 'validation_fraction':0.1, 'verbose':False,
                    'warm_start':False},
            "ET": {'n_estimators': 50, 'max_depth': 15, 'n_jobs': -1, 'min_samples_split': 10, 'min_samples_leaf': 5},
            "ADA": {'base_estimator': None, 'n_estimators': 100, 'learning_rate': 1.25, 'algorithm': 'SAMME.R', 'random_state': self.SEED},
            "GBM": {'loss':'deviance', 'learning_rate': 0.1, 'n_estimators': 50, 
                   'subsample': 1.0, 'criterion': 'friedman_mse', 'min_samples_split': 5, 'min_samples_leaf': 5, 
                   'min_weight_fraction_leaf': 0.0, 'max_depth': 5, 'min_impurity_split': 1e-07, 'init': None, 
                   'random_state': None, 'max_features': None, 'verbose': 0, 'max_leaf_nodes': None, 
                   'warm_start': False, 'presort': 'auto'},
            "LR": {'penalty':'l2', 'dual': False, 'tol':0.0001, 'C':0.9},
            "XGB": {'seed': self.SEED, 'n_estimators': 100, 'max_depth': 3,
                    'learning_rate': 0.1, 'objective': 'reg:linear', 'nthread': -1}, # seperate lib, XGBOOST
            "RVM": {}, # seperate code, RVM
            "DNN": {}, # deep network (dense fully connected layers)
            "CNN": {'architecture': None, 'model_location': None}, # convolutional network , 
                    #architecture: vgg16, vgg19, resnet50, inception, xception
            "EBE": {}, # custom predictor for low sample/high dimensional data
            "LGBM": {'boosting_type':'gbdt' ,'learning_rate': 0.75, # boosting: gbdt,dart,goss,rf
                    'max_depth': 4, 'num_leaves': 100, 'n_jobs': -1,
                    'n_estimators':100, 'random_state': self.SEED},
            "CART":{'criterion':'gini', 'splitter':'best', 
                    'max_depth':10, 'min_samples_split':10, 'min_samples_leaf':5, 
                    'min_weight_fraction_leaf':0.0, 'max_features': None, 
                    'random_state': self.SEED, 'max_leaf_nodes': None, 
                    'min_impurity_split':1e-07, 'class_weight':None, 
                    'presort':False},
            "GPC":{'optimizer': 'fmin_l_bfgs_b', 'n_restarts_optimizer': 0, 
                    'max_iter_predict': 100, 'warm_start': False, 
                    'copy_X_train': True, 'random_state': self.SEED, 
                    'multi_class': 'one_vs_rest', 'n_jobs': 1},
            "LDA":{'shrinkage': 'auto', 'solver': 'lsqr', 'priors': None},
            "QDA":{'priors': None, 'reg_param': 0.0, 'store_covariance':False, 'tol':0.0001, 'store_covariances':None}
        }
        #http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
        self.FEATURE_SELECTION_PARAMETERS = {   "low_variance":{"lib": "sklearn"}, 
                                                "RFECV":{"lib": "sklearn"}, 
                                                "RFE":{"lib": "sklearn"},
                                                "univariate":{"lib": "sklearn"}
                                            }
        self.DIMENSION_REDUCTION_PARAMETERS = {"pca": {'copy' : True, 'whiten' : False, 
                                                       'svd_solver' : 'full', 
                                                       'tol' : 0.001, 'iterated_power':'auto', 
                                                       'random_state' : None},
                                                "pls": {'scale': True, 
                                                        'max_iter':500, 
                                                        'tol': 1e-06, 
                                                        'copy':True},
                                                "lda":{'solver':'svd', 
                                                        'shrinkage':None,
                                                        'priors':None,
                                                        'store_covariance':False,
                                                        'tol':0.0001},
                                                "rbm":{'random_state': 0, 
                                                       'verbose': True,
                                                        'n_iter': 100,
                                                        'learning_rate': 0.01},
                                                "t-SNE":{'n_components':3, 
                                                         'perplexity': 30.0, 
                                                         'early_exaggeration': 12.0, 
                                                         'learning_rate':200.0, 
                                                         'n_iter':1000, 
                                                         'n_iter_without_progress':300, 
                                                         'min_grad_norm':1e-07, 
                                                         'metric':'euclidean', 
                                                         'init':'random', 
                                                         'verbose':0, 
                                                         'random_state':None, 
                                                         'method':'barnes_hut',
                                                         'angle':0.5},
                                                "lle": {'n_neighbors':5, 
                                                        'n_components':3, 
                                                        'reg':0.001,
                                                        'eigen_solver':'auto', 
                                                        'tol':1e-06, 
                                                        'max_iter':100, 
                                                        'method':'standard', 
                                                        'hessian_tol':0.0001, 
                                                        'modified_tol':1e-12, 
                                                        'neighbors_algorithm':'auto', 
                                                        'random_state':None, 
                                                        'n_jobs':4},
                                                "mds": {'n_components':3, 
                                                        'metric':True, 
                                                        'n_init':4, 
                                                        'max_iter':300, 
                                                        'verbose':0, 
                                                        'eps':0.001, 
                                                        'n_jobs':1, 
                                                        'random_state':None, 
                                                        'dissimilarity':'euclidean'},
                                                "isomap": { 'n_neighbors':5, 
                                                            'n_components':3, 
                                                            'eigen_solver':'auto', 
                                                            'tol':0, 
                                                            'max_iter':None, 
                                                            'path_method':'auto', 
                                                            'neighbors_algorithm':'auto', 
                                                            'n_jobs':4},
                                                "ica":{'algorithm':'parallel', 
                                                        'whiten': True, 
                                                        'fun':'logcosh', 
                                                        'fun_args':None, 
                                                        'max_iter':200, 
                                                        'tol': 0.0001, 
                                                        'w_init':None, 
                                                        'random_state':None, 
                                                        'return_X_mean':False, 
                                                        #'compute_sources':True, 
                                                        'return_n_iter':False
                                                        },
                                                "sae": {"layers": [2000, 1000, 500]}
                                            }
        self.PIPELINE_PARAMETERS =   {  "scaler": {"type": "minmax"},
                                        "pre_processing": {"patient_grouping": 'mean', "bias_removal": False}, # patient grouping and cohort bias removal
                                        "dim_reduction": {"type": "PCA", "n_comp": 1000},
                                        "feature_selection": {"type": "low_variance", 
                                                              "p_value": 0.05, 
                                                              "stat_method": 'wilcoxon'}}

    def _read_cohort(self, path):
        if self.SET_NAME == 'ALL_10':
            ch1 = pd.read_csv(path, sep="\t")
            patient_ids = ch1.columns.values[1:]
            patient_ids = [pid.split(".")[0] + ".CEL" for pid in patient_ids]

            gene_ids = ch1.ix[:,0]
            #
            ch1_m = ch1.values[:,1:].T
            ch1 = pd.DataFrame(data=ch1_m,index=patient_ids,columns=gene_ids, dtype=float)
            labels = list(ch1.filter(axis=1, regex=r"^(AFFX.*)").columns)
            ch1 = ch1.drop(labels, axis=1)
            #ch1['ID'] = ch1.index
        elif self.SET_NAME == 'MELA': # assumes NCBI format, assumes first row of target contains actual targets..
            ch1 = pd.read_csv(path, sep="\t", skiprows=self.READ_PARAMETERS['header_rows'], skipfooter=1, engine='python')
            patient_ids = ch1.loc[ch1.ix[:,0]==self.READ_PARAMETERS['ID']].ix[:,1:].values
            #targets = ch1.loc[ch1.ix[:,0]==read_dict['target']].reset_index(drop=True).loc[0,:][1:]
            target_col = ch1.loc[ch1.ix[:,0]==self.READ_PARAMETERS['target']].values.T[1:,0]
            gene_ids = ch1.ix[(self.READ_PARAMETERS['genome_line_range'][0]-self.READ_PARAMETERS['header_rows']-2):self.READ_PARAMETERS['genome_line_range'][1],0].values
            #gene_ids = np.append(gene_ids_,'target')
            ch1_m = ch1.values[list(range(self.READ_PARAMETERS['genome_line_range'][0]-self.READ_PARAMETERS['header_rows']-2, 
                                          self.READ_PARAMETERS['genome_line_range'][1]-self.READ_PARAMETERS['header_rows']-1)),1:].T
            ch1 = pd.DataFrame(data=ch1_m, index=patient_ids[0,:], columns=gene_ids, dtype=float)
            ch1['target'] = pd.Series(target_col, index=ch1.index)
            ch1['ID'] = ch1.index
            ch1 = ch1.sample(frac=1)#.reset_index(drop=True)
            ch1.index = ch1['ID']
            labels = list(ch1.filter(axis=1, regex=r"^(AFFX.*)").columns)
            ch1 = ch1.drop(labels, axis=1)
        return ch1

    def _read_patient_file(self, path):
        patients = pd.read_excel(path)
        #columns = patients.ix[0].values
        #patients = patients.drop(patients.index[0])
        #patients.columns = columns

        return patients

    def _read_modelling_data(self):
        dat = pd.read_pickle(self.DATA_loc)
        return dat

    def load_probeset_data(self):
        # ch1 = read_cohort("Data/cohort1_plus2.txt")
        # ch2 = read_cohort("Data/cohort2_plus2.txt")
        # cha = read_cohort("Data/cohortALL10_plus2.txt")
        if(self.SET_NAME == 'ALL_10'):
            self.MODEL_PARAMETERS['target'] = 'Treatment_risk_group_in_ALL10'
            self.MODEL_PARAMETERS['ID'] = 'labnr_patient'
            self.DATA_all_samples = self._read_cohort("_data/genomic_data/leukemia/all_10.txt")
            self.DATA_patients = self._read_patient_file("_data/genomic_data/patients.xlsx")
            self.DATA_Tnormal = pd.read_csv("_data/genomic_data/TALLnormalTcellsTransposed.txt", sep="\t")            

            self.DATA_patients = self.DATA_patients[(self.DATA_patients.Age<17) or (np.isnan(self.DATA_patients.Age))]

            if(self.DATA_loc is None):
                self.DATA_merged = pd.merge(self.DATA_patients, 
                                        self.DATA_all_samples, 
                                        how='left', left_on="Microarray file", right_index=True)
            else:
                self.DATA_merged = self._read_modelling_data()

            if(self.write_out == True):
                self.DATA_merged.to_pickle("_data/genomic_data/data_ALL10.pkl")

            self.DATA_merged['WhiteBloodCellcount']= pd.to_numeric(self.DATA_merged['WhiteBloodCellcount'])

            if (self.DEBUG == True): # reduced number of genomes to run through code logic more quickly
                self.DATA_merged = self.DATA_merged[self.DATA_merged.columns[:10000]]
        elif(self.SET_NAME =='MELA'):
            # target : !Sample_characteristics_ch1
            # ID : ID_REF
            # line 64>= for genomic expressions
            self.MODEL_PARAMETERS['target'] = 'target'
            self.MODEL_PARAMETERS['ID'] = 'ID'
            self.READ_PARAMETERS = {"target": "!Sample_characteristics_ch1", "ID": "ID_REF", "genome_line_range": [64, 22346], "header_rows": 28}
            self.DATA_merged = self._read_cohort("_data/genomic_data/melanoma/mela.txt")            
        return self.DATA_merged
        

    from functions.get_predictors import classify_treatment, run_classification

    def main():
        load_probeset_data()


if __name__ == '__main__':
    Rocket = RexR(datalocation = None, #'_data/genomic_data/data.pkl', 
                   seed = 3123, 
                   debug = True, 
                   write_out=True,
                   set_name = 'ALL_10') # data to read in ALL_10, or MELA
    data = Rocket.load_probeset_data()