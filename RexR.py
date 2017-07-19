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



'''




class RexR():
    # constants
    DATA_all_samples = None
    DATA_patients = None
    DATA_Tnormal = None
    DATA_merged = None
    DATA_merged_processed = None
    DATA_loc = '/home/bramiozo/DEV/RexR/_data/genomic_data/data.pkl'
    SEED = 1234
    debug = False

    def __init__(self, datalocation = '/home/bramiozo/DEV/RexR/_data/genomic_data/data.pkl', seed = 2412, debug = False):
        print("+"*30, 'Firing up RexR!', "+"*30)
        self.DATA_loc = datalocation
        self.SEED = seed
        self.DEBUG = debug



    def _read_cohort(self, path):
        ch1 = pd.read_csv(path, sep="\t")
        patient_ids = ch1.columns.values[1:]
        patient_ids = [pid.split(".")[0] + ".CEL" for pid in patient_ids]

        gene_ids = ch1.ix[:,0]

        ch1_m = ch1.values[:,1:].T
        ch1 = pd.DataFrame(data=ch1_m,index=patient_ids,columns=gene_ids)

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

    def load_probeset_data(self, write_out = False, read_in = False):
        # ch1 = read_cohort("Data/cohort1_plus2.txt")
        # ch2 = read_cohort("Data/cohort2_plus2.txt")
        # cha = read_cohort("Data/cohortALL10_plus2.txt")


        self.DATA_all_samples = self._read_cohort("_data/genomic_data/all_samples.txt")
        self.DATA_patients = self._read_patient_file("_data/genomic_data/patients.xlsx")
        self.DATA_Tnormal = pd.read_csv("_data/genomic_data/TALLnormalTcellsTransposed.txt", sep="\t")
   
        all_10 = ["ALL-10","IA","JB"]

        if(read_in == False):
            self.DATA_merged = pd.merge(self.DATA_patients, 
                                    self.DATA_all_samples, 
                                    how='left', left_on="Microarray file", right_index=True)
        else:
            self.DATA_merged = self._read_modelling_data()

        if(write_out == True):
            self.DATA_merged.to_pickle("_data/genomic_data/data.pkl")

        self.DATA_merged['WhiteBloodCellcount']= pandas.to_numeric(self.DATA_merged['WhiteBloodCellcount'])

        if (self.DEBUG == True): # reduced number of genomes to run through code logic more quickly
            self.DATA_merged = self.DATA_merged[self.DATA_merged.columns[:10000]]



        return self.DATA_merged
        

    from functions.get_predictors import classify_treatment, get_top_genes

    def main():
        load_probeset_data()


if __name__ == '__main__':
        main()
