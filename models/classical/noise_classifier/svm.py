from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from load_dataset import load_dataset

def train_svm(X_train, y_train, X_test, y_test):
    
    # Step 1: Encode string labels to integers
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    # Step 2: Create pipeline with imputation + standardization + SVM
    clf = make_pipeline(
        SimpleImputer(strategy='mean'),   # Handle NaNs by filling with column means
        StandardScaler(),
        SVC(kernel='rbf', probability=True)
    )

    # Step 3: Train the model
    clf.fit(X_train, y_train_enc)

    # Step 4: Predict and decode labels back
    y_pred_enc = clf.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_enc)

    # Step 5: Evaluation
    print(classification_report(y_test, y_pred))
    return clf


X_train, y_train = load_dataset("/content/balanced_data/train")
X_test, y_test = load_dataset("/content/balanced_data/test")
train_svm(X_train, y_train, X_test, y_test)