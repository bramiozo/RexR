from sklearn import preprocessing, model_selection

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
        var_columns = df.columns[12:15]

    train_idx = df[target].isin(["HR","MR","SR"])

    y = df[train_idx][target].map(lambda x: 0 if x in ["MR", "SR"] else 1).values
    x = df.loc[train_idx,var_columns].values

    return x,y

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