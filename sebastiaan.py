import pandas as pd
from collections import Counter 
import numpy as np
from sklearn import preprocessing, svm, model_selection, metrics


####
# AUTHORS
# Sebastiaan de Jong
# Wybe Rozema
# Bram van Es
# Sabrina Wandl
# Nick Hales

def read_cohort(path):
    ch1 = pd.read_csv(path, sep="\t")
    patient_ids = ch1.columns.values[1:]
    patient_ids = [pid.split(".")[0] + ".CEL" for pid in patient_ids]

    gene_ids = ch1.ix[:,0]

    ch1_m = ch1.values[:,1:].T
    ch1 = pd.DataFrame(data=ch1_m,index=patient_ids,columns=gene_ids)

    return ch1

def read_patient_file(path):
    patients = pd.read_excel(path)
    columns = patients.ix[0].values
    patients = patients.drop(patients.index[0])
    patients.columns = columns

    return patients

def load_data():
    # ch1 = read_cohort("Data/cohort1_plus2.txt")
    # ch2 = read_cohort("Data/cohort2_plus2.txt")
    # cha = read_cohort("Data/cohortALL10_plus2.txt")
    all_samples = read_cohort("Data/all_samples.txt")
    # normal_samples = read_cohort("Data/TALLnormal.txt")

    patients = read_patient_file("Data/patients.xlsx")

    df = pd.merge(patients, all_samples, how='left',\
        left_on="Microarray file", right_index=True)
    # df = pd.merge(df, normal_samples, how='left',\
    #   left_on="Microarray file", right_index=True)

    df.to_pickle("data.pickle")

def preprocess(df):
    gene_columns = df.columns[21:]

    scaler = preprocessing.StandardScaler()
    ch1 = df["array-batch"].isin(["cohort 1"])
    ch2 = df["array-batch"].isin(["cohort 2"])
    cha = df["array-batch"].isin(["JB", "IA", "ALL-10"])

    df.loc[ch1,gene_columns] = scaler.fit_transform(df.loc[ch1,gene_columns])
    df.loc[ch2,gene_columns] = scaler.fit_transform(df.loc[ch2,gene_columns])
    df.loc[cha,gene_columns] = scaler.fit_transform(df.loc[cha,gene_columns])

    return df

def group_patients(df):
    # should try to combine expression profiles instead
    df = df.groupby("labnr patient").apply(lambda g:g.iloc[0])
    return df 

def get_matrix(df): 
    gene_columns = df.columns[21:]

    target = 'Treatment risk group in ALL10'
    train_idx = df[target].isin(["HR","MR","SR"])

    y = df[train_idx][target].map(lambda x: 0 if x in ["MR", "SR"] else 1).values
    x = df.loc[train_idx,gene_columns].values

    return x,y

def classify_treatment(x,y):
    model = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0)

    splitter = model_selection.StratifiedKFold(n_splits=e)

    for train_index, test_index in splitter.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index] 

        print(train_index.shape, test_index.shape)
        model.fit(x_train,y_train)

        pred = model.predict(x_test)

        print(metrics.accuracy_score(y_test,pred))

def main():
    # load_data()
    df = pd.read_pickle("data.pickle")
    df = preprocess(df)
    df = group_patients(df)
    x,y = get_matrix(df)
    classify_treatment(x,y)

if __name__ == '__main__':
    main()
