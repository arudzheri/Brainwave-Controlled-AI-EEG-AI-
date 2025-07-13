import pandas as pd
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def load_data():
    files = glob.glob('data/*.csv')
    dfs = []
    for file in files:
        df = pd.read_csv(file, header=None)
        dfs.append(df)
    data = pd.concat(dfs)
    X = data.iloc[:, 1:]  # features
    y = data.iloc[:, 0]   # labels
    return X, y

def train():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    print(classification_report(y_test, clf.predict(X_test)))
    joblib.dump(clf, 'models/eeg_model.pkl')

if __name__ == "__main__":
    train()

