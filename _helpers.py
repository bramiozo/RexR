from sklearn import preprocessing, svm, tree, ensemble, naive_bayes, neural_network, model_selection, metrics
import numpy as np

def _group_patients(df, method = 'first'): # method = ['first', 'average', 'median', 'min', 'max']
    # 1. clean probesets 
    # 2. reduce multiple probesets per gene to one  
    if(method == 'first'):
        df = df.groupby("labnr patient").apply(lambda g: g.iloc[0])
    elif(method == 'average'):
        df = df.groupby("labnr patient").mean()
    elif(method == 'median'):
        df = df.groupby("labnr patient").median()
    elif(method == 'min'):
        df = df.groupby("labnr patient").min()
    elif(method == 'max'):
        df = df.groupby("labnr patient").max()

    return df 

def _get_matrix(df, type = 'genomic', target = 'Treatment risk group in ALL10'): # type = ['genomic', ] 
    if(type=='genomic'):
        var_columns = df.columns[21:]
    elif(type=='patient'):
        var_columns = ["Age", "WhiteBloodCellcount", "Gender"]
        df[var_columns] = df[var_columns].fillna(0.0)

    train_idx = df[target].isin(["HR","MR","SR"])

    y = df[train_idx][target].map(lambda x: 0 if x in ["MR", "SR"] else 1).values
    x = df.loc[train_idx,var_columns].values

    return x,y

def _survival_matrix(df):
    valid = [0,1]
    gene_columns = df.columns[21:]

    target = "code_OS"

    df = df[df[target].isin(valid)]

    return df[gene_columns].values, df[target].values

def _preprocess(df):
    gene_columns = df.columns[21:]

    scaler = preprocessing.StandardScaler()
    ch1 = df["array-batch"].isin(["cohort 1"])
    ch2 = df["array-batch"].isin(["cohort 2"])
    cha = df["array-batch"].isin(["JB", "IA", "ALL-10"])

    df.loc[ch1,gene_columns] = scaler.fit_transform(df.loc[ch1,gene_columns])
    df.loc[ch2,gene_columns] = scaler.fit_transform(df.loc[ch2,gene_columns])
    df.loc[cha,gene_columns] = scaler.fit_transform(df.loc[cha,gene_columns])

    return df

def _benchmark_classifier(model, x, y, splitter, seed):
    splitter.random_state = seed
    pred = np.zeros(shape=y.shape)
    coef = np.zeros(shape=(1, x.shape[1]))

    for train_index, test_index in splitter.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index] 

        model[1].fit(x_train,y_train)
        pred[test_index] = model[1].predict(x_test)
        # coef += model.coef_
        

    return pred