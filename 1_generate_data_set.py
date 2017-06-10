import pandas as pd
from collections import Counter 
import numpy as np

def read_cohort(path):
	ch1 = pd.read_csv(path, sep="\t")
	patient_ids = ch1.columns.values[1:]
	patient_ids = [pid.split(".")[0] + ".CEL" for pid in patient_ids]

	gene_ids = ch1.ix[:,0]

	ch1_m = ch1.values[:,1:].T
	ch1 = pd.DataFrame(data=ch1_m,index=patient_ids,columns=gene_ids)

	return ch1

def load_data():
	# ch1 = read_cohort("Data/cohort1_plus2.txt")
	# ch2 = read_cohort("Data/cohort2_plus2.txt")
	# cha = read_cohort("Data/cohortALL10_plus2.txt")
	all_samples = read_cohort("/home/wybe/hackathon/raw/all_samples.txt")

	patients = pd.read_excel("/home/wybe/hackathon/raw/patients.xlsx")
	columns = patients.ix[0].values
	patients = patients.drop(patients.index[0])
	patients.columns = columns

	all_10 = ["ALL-10","IA","JB"]

	join = pd.merge(patients, all_samples, how='left', left_on="Microarray file", right_index=True)
	null = join.isnull().sum()
	# print(all_samples.head(10))
	# print(null[null < 343])

	print(join[join["Treatment protocol"] == "ALL10"])

	return join[join["Treatment protocol"] == "ALL10"]
	# print(len(patients.columns))
	# print(patients.a(10))

def write_data(dat_set):

	#write as csv
	dat_set.to_csv("/home/wybe/hackathon/raw/modelling_dataset.txt")

def main():
	dat_set = load_data()

	write_data(dat_set)

if __name__ == '__main__':
	main()

