from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def train_svm(X_train, y_train):
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True))
    clf.fit(X_train, y_train)
    return clf
