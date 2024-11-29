# In algo.py
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


def svc(X, y):
    """SVC model training and evaluation."""

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train SVC
    svc_model = SVC(kernel='rbf', C=1.0)
    svc_model.fit(X_train, y_train)

    # Predict
    y_pred = svc_model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    

    return {"model": svc_model, "accuracy": accuracy, "report": report}
