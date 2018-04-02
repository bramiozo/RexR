import pandas as pd 
import numpy as np 

from sklearn import ensemble, model_selection, metrics, base, feature_selection, pipeline, preprocessing

def load_data(path):
    raw_df = pd.read_csv(path, sep="\t", comment="!").transpose()
    columns = raw_df.iloc[0]
    index = raw_df.index[1:]
    data = raw_df.iloc[1:,0:].values
    df = pd.DataFrame(data=data, index=index,columns=columns)
    df = df.astype(float)

    return df

def parse_metadata(path, line_indicator="!"):
    lines = []
    with open(path, "r") as f:
        for l in f:
            l = l.strip()
            if len(l) == 0:
                pass
            elif l[0] == line_indicator:
                lines.append(l)
            else:
                break 

    values = list(map(lambda l: l.split("\t"), lines))
    keys = [x[0][1:] for x in values]
    values = [x[1:] for x in values]

    metadata = pd.DataFrame({"key":keys,"value":values})
        
    return metadata

def make_targets(df, sample_info, drop_threshold=4):
    df['class_name'] = sample_info
    df = df.groupby('class_name').filter(lambda x: len(x) > drop_threshold)
    classes = df['class_name'].unique()
    df['target'] = df['class_name'].replace(dict(zip(classes, range(len(classes)))))

    return df

def scale(x):
    return x

def benchmark(x,y,clf):
    cv = model_selection.KFold(n_splits=10, random_state=1)

    predictions = np.zeros(shape=y.shape)

    fitted = []

    for train_index, test_index in cv.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = base.clone(clf)
        clf.fit(x_train, y_train)
        predictions[test_index] = clf.predict(x_test)
        fitted.append(clf)

    return fitted,predictions

def main():
    gse7553_path = "GSE7553_series_matrix.txt"
    gse15605_path = "GSE15605_series_matrix.txt"

    df = load_data(gse15605_path)
    metadata = parse_metadata(gse15605_path)
    sample_descr = metadata[metadata.key == "Sample_title"].value.values[0]
    # sample_labels = [x.split(",")[1].strip() for x in sample_descr]
    sample_labels = [x.strip("\"").split(" ")[0] for x in sample_descr]
    df = make_targets(df, sample_labels)

    y = df['target'].values
    x = df.loc[:, ~df.columns.isin(['target', 'class_name'])].values

    pipe = pipeline.Pipeline([
        ("scaler", preprocessing.MinMaxScaler(feature_range=(0,1))),
        ("variance", feature_selection.VarianceThreshold(threshold=0.05)),
        ("model", ensemble.RandomForestClassifier(n_estimators=200))
    ])

    _, y_pred = benchmark(x,y,pipe)

    print(metrics.confusion_matrix(y,y_pred))
    print(metrics.accuracy_score(y,y_pred))


if __name__ == '__main__':
    main()