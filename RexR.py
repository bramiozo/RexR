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



class RexR():
    # constants
    DATA_all_samples = None
    DATA_patients = None
    DATA_TALL_Tnormal = None
    DATA_merged = None

    def __init__(self, message = 'Firing up RexR'):
        print(message)

    def read_cohort(self, path):
        ch1 = pd.read_csv(path, sep="\t")
        patient_ids = ch1.columns.values[1:]
        patient_ids = [pid.split(".")[0] + ".CEL" for pid in patient_ids]

        gene_ids = ch1.ix[:,0]

        ch1_m = ch1.values[:,1:].T
        ch1 = pd.DataFrame(data=ch1_m,index=patient_ids,columns=gene_ids)

        return ch1

    def load_probeset_data(self, write_out=False):
        # ch1 = read_cohort("Data/cohort1_plus2.txt")
        # ch2 = read_cohort("Data/cohort2_plus2.txt")
        # cha = read_cohort("Data/cohortALL10_plus2.txt")
        self.DATA_all_samples = self.read_cohort("_data/genomic_data/all_samples.txt")
        self.DATA_patients = pd.read_excel("_data/genomic_data/patients.xlsx")
        self.TALL_Tnormal = pd.read_csv("_data/genomic_data/TALLnormalTcellsTransposed.txt")
        columns = self.DATA_patients.ix[0].values
        self.DATA_patients = self.DATA_patients.drop(self.DATA_patients.index[0])
        self.DATA_patients.columns = columns

        all_10 = ["ALL-10","IA","JB"]

        self.DATA_merged = pd.merge(self.DATA_patients, 
                                    self.DATA_all_samples, how='left', left_on="Microarray file", right_index=True)
        if(write_out == True):
            self.DATA_merged.to_csv("_data/genomic_data/modelling_dataset.txt")

        return self.DATA_merged
        
        # null = join.isnull().sum()
        # print(all_samples.head(10))
        # print(null[null < 343])

        # print(join[join["Treatment protocol"] == "ALL10"])


        # print(len(patients.columns))
        # print(patients.a(10))

    #def write_data(dat_set):
    #    #write as csv
    #    dat_set.to_csv("_data/genomic_data/modelling_dataset.txt")

    from functions.get_principal_components import get_principal_components


    def main():
        load_probeset_data()


if __name__ == '__main__':
        main()
