def _group_patients(self, df):
    # 1. clean probesets 
    # 2. reduce multiple probesets per gene to one  
    df = df.groupby("labnr patient").apply(lambda g:g.iloc[0])

    return df 

def _get_matrix(self, df): 
    gene_columns = df.columns[21:]

    target = 'Treatment risk group in ALL10'
    train_idx = df[target].isin(["HR","MR","SR"])

    y = df[train_idx][target].map(lambda x: 0 if x in ["MR", "SR"] else 1).values
    x = df.loc[train_idx,gene_columns].values

    return x,y

def _preprocess(self, df):
    gene_columns = df.columns[21:]

    scaler = preprocessing.StandardScaler()
    ch1 = df["array-batch"].isin(["cohort 1"])
    ch2 = df["array-batch"].isin(["cohort 2"])
    cha = df["array-batch"].isin(["JB", "IA", "ALL-10"])

    df.loc[ch1,gene_columns] = scaler.fit_transform(df.loc[ch1,gene_columns])
    df.loc[ch2,gene_columns] = scaler.fit_transform(df.loc[ch2,gene_columns])
    df.loc[cha,gene_columns] = scaler.fit_transform(df.loc[cha,gene_columns])

    return df