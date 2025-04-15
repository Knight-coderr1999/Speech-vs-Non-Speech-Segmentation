from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from load_dataset import load_dataset

X_train, y_train = load_dataset("/content/balanced_data/train")
X_test, y_test = load_dataset("/content/balanced_data/test")

clf = RandomForestClassifier(n_estimators=200)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
