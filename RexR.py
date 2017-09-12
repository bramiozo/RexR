import pandas as pd
from collections import Counter 
import numpy as np
import pandas


####
# AUTHORS
# Sebastiaan de Jong
# Wybe Rozema
# Bram van Es
# Sabrina Wandl
# Nick Heuls

# Jules Meijerink
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

to-do's (september 2017):
- deep learner
- sparse auto encoding
- t-sne / optics analyser
- ROC/confusion matrix visualiser
- patient similarity
- xgboost
- cohort-bias reducer
- conditional survival estimator

'''




class RexR():
    # constants
    DATA_all_samples = None
    DATA_patients = None
    DATA_Tnormal = None
    DATA_merged = None
    DATA_merged_processed = None
    DATA_loc = None
    MODEL_PARAMETERS = {}




    write_out = None
    SEED = 1234
    debug = False

    def __init__(self, datalocation = None, seed = 2412, debug = False, write_out = False):
        print("+"*30, 'Firing up RexR!', "+"*30)
        self.DATA_loc = datalocation
        self.SEED = seed
        self.DEBUG = debug
        self.write_out = write_out
        self.MODEL_PARAMETERS = {
            "SVM":{'degree': 3, 'tol': 0.0001, 'C':  0.9, 'probability' : True},
            "RF": {'n_estimators': 100, 'max_depth': 25, 'n_jobs': -1, 'min_samples_split': 5, 'min_samples_leaf': 5},
            "MLNN": {'activation':'tanh', 'alpha':1e-05, 'batch_size':'auto',
                    'beta_1':0.9, 'beta_2':0.999, 'early_stopping':False,
                    'epsilon':1e-08, 'hidden_layer_sizes':(60, 30, 15, 7, 2), 'learning_rate':'adaptive',
                    'learning_rate_init':0.001, 'max_iter':200, 'momentum':0.9,
                    'nesterovs_momentum':True, 'power_t':0.5, 'random_state':1, 'shuffle':True,
                    'solver':'adam', 'tol':0.0001, 'validation_fraction':0.1, 'verbose':False,
                    'warm_start':False},
            "ET": {'n_estimators': 100, 'max_depth': 25, 'n_jobs': -1, 'min_samples_split': 5, 'min_samples_leaf': 5},
            "ADA": {'base_estimator': None, 'n_estimators': 150, 'learning_rate': 1.0, 'algorithm': 'SAMME.R', 'random_state': self.SEED},
            "GBM": {'loss':'deviance', 'learning_rate': 0.1, 'n_estimators': 100, 
                   'subsample': 1.0, 'criterion': 'friedman_mse', 'min_samples_split': 5, 'min_samples_leaf': 5, 
                   'min_weight_fraction_leaf': 0.0, 'max_depth': 4, 'min_impurity_split': 1e-07, 'init': None, 
                   'random_state': None, 'max_features': None, 'verbose': 0, 'max_leaf_nodes': None, 
                   'warm_start': False, 'presort': 'auto'},
            "LR": {'penalty':'l2', 'dual': False, 'tol':0.0001, 'C':0.9},
            "XGB": {},
            "RVM": {},
            "DNN": {},
            "CNN": {},
            "CART":{'criterion':'gini', 'splitter':'best', 
                    'max_depth':10, 'min_samples_split':2, 'min_samples_leaf':3, 
                    'min_weight_fraction_leaf':0.0, 'max_features': None, 
                    'random_state': self.SEED, 'max_leaf_nodes': None, 
                    'min_impurity_split':1e-07, 'class_weight':None, 
                    'presort':False},
            "GPC":{'optimizer': 'fmin_l_bfgs_b', 'n_restarts_optimizer': 0, 
                    'max_iter_predict': 100, 'warm_start': False, 
                    'copy_X_train': True, 'random_state': None, 
                    'multi_class': 'one_vs_rest', 'n_jobs': 1}
        }


    def _read_cohort(self, path):
        ch1 = pd.read_csv(path, sep="\t")
        patient_ids = ch1.columns.values[1:]
        patient_ids = [pid.split(".")[0] + ".CEL" for pid in patient_ids]

        gene_ids = ch1.ix[:,0]

        ch1_m = ch1.values[:,1:].T
        ch1 = pd.DataFrame(data=ch1_m,index=patient_ids,columns=gene_ids, dtype=float)

        return ch1

    def _read_patient_file(self, path):
        patients = pd.read_excel(path)
        columns = patients.ix[0].values
        patients = patients.drop(patients.index[0])
        patients.columns = columns

        return patients

    def _read_modelling_data(self):
        dat = pd.read_pickle(self.DATA_loc)
        return dat

    def load_probeset_data(self):
        # ch1 = read_cohort("Data/cohort1_plus2.txt")
        # ch2 = read_cohort("Data/cohort2_plus2.txt")
        # cha = read_cohort("Data/cohortALL10_plus2.txt")

        self.DATA_all_samples = self._read_cohort("_data/genomic_data/all_samples.txt")
        self.DATA_patients = self._read_patient_file("_data/genomic_data/patients.xlsx")
        self.DATA_Tnormal = pd.read_csv("_data/genomic_data/TALLnormalTcellsTransposed.txt", sep="\t")
   
        all_10 = ["ALL-10","IA","JB"]

        if(self.DATA_loc is None):
            self.DATA_merged = pd.merge(self.DATA_patients, 
                                    self.DATA_all_samples, 
                                    how='left', left_on="Microarray file", right_index=True)
        else:
            self.DATA_merged = self._read_modelling_data()

        if(self.write_out == True):
            self.DATA_merged.to_pickle("_data/genomic_data/data.pkl")

        self.DATA_merged['WhiteBloodCellcount']= pandas.to_numeric(self.DATA_merged['WhiteBloodCellcount'])

        if (self.DEBUG == True): # reduced number of genomes to run through code logic more quickly
            self.DATA_merged = self.DATA_merged[self.DATA_merged.columns[:10000]]



        return self.DATA_merged
        

    from functions.get_predictors import classify_treatment

    def main():
        load_probeset_data()


if __name__ == '__main__':
        main()
