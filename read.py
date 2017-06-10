# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
os.getcwd()
os.chdir(r'\\solon.prd\branches\P\Global\Users\C20207\Userdata\Desktop\hackathon\Hackaton')

import pandas

import re

# Usage:
file_allsamples = "all samples_plus2.txt"
data_allsamples = pandas.DataFrame.from_csv(file_allsamples, sep='\t')
#dict_allsamples = create_dictionary(file_allsamples)
#file_cohort1 = "cohort1_plus2.txt"
#data_cohort1= pandas.DataFrame.from_csv(file_cohort1, sep='\t')

#file_cohort2 = "cohort2_plus2.txt"
#data_cohort2 = pandas.DataFrame.from_csv(file_cohort2, sep='\t')

#file_cohortAll10 = "cohortALL10_plus2.txt"
#data_cohortAll10 = pandas.DataFrame.from_csv(file_cohortAll10, sep='\t')

file_HG = "HG-U133_Plus_2.na36annotatie.txt"
data_HG = pandas.DataFrame.from_csv(file_HG, sep='\t')

file_HT = "HT_HG-U133_Plus_PM.na35.annot.txt" 
data_HT = pandas.DataFrame.from_csv(file_HT, sep='\t')

file_normal = "T ALL normal Tcells transposed.txt" 
data_normal = pandas.DataFrame.from_csv(file_normal, sep='\t')

file_sample = "Sample Description Dataset T-ALL details hackaton-3.xlsx" 
data_sample = pandas.read_excel(file_sample, sep='\t')

data_sample_treatment = data_sample.loc[data_sample['Treatment risk group in ALL10'].isin(['HR_eq','HR','SR','MR'])]

del data_sample_treatment['Tissue-Disease']
del data_sample_treatment['Study group']
del data_sample_treatment['Treatment protocol']

#klant ID sourcen
columnID=data_allsamples.columns.tolist()
#pattern uithalen: _iiii of iii(i).
clientID=columnID
re.match("\_[\d]{3,4} | ^[\d]{3,4}\.",columnID)

#data transponeren
data_allsamples_trans=data_allsamples.transpose()
#klant id toevoegen aan data_allsamples

#datasets mergen
pandas.merge(data_allsamples,data_sample_treatment,on='ID',how='outer')

clientID2=data_sample_treatment.columns.tolist()

