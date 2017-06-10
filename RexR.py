import pandas as pd
from collections import Counter 
import numpy as np


####
# AUTHORS
# Sebastiaan de Jong
# Wybe Rozema
# Bram van Es
# Sabrina Wandl
# Nick Heuls
#################


class RexR():
    # constants
    DATA_all_samples = None
    DATA_patients = None
    DATA_Tnormal = None
    DATA_merged = None
    DATA_loc = '/home/bramiozo/DEV/RexR/_data/genomic_data/data.pkl'

    def __init__(self, datalocation = '/home/bramiozo/DEV/RexR/_data/genomic_data/data.pkl', message = 'Firing up RexR!'):
        print(message)
        self.DATA_loc = datalocation


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

        return self.DATA_merged
        

    from functions.get_dimension_reduction import get_principal_components, get_linear_discriminant_analysis, get_quadrant_discriminant_analysis, get_vector_characteristics
    from functions.get_predictors import classify_treatment_model, estimate_survival

    def main():
        load_probeset_data()
        

if __name__ == '__main__':
        main()
