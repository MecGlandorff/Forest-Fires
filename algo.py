

import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def svr(data):

    """ Everything SVR ;)"""

    # We drop 'area' from the X dataframe, since area is the value we want to predict and make it Y
    X = data.drop(columns=['area'])
    y = data['area']

    # Split 80/20, train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale x and y so the values are in range 0,1
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

    # Run the SVR and train it
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr.fit(X_train, y_train)

    y_pred = svr.predict(X_test)

    y_pred_orig = scaler_y.inverse_transform(y_pred)
    y_test_orig = scaler_y.inverse_transform(y_test)

    # Results MSE and R-squared
    mse = mean_squared_error(y_test_orig, y_pred_orig)
    r2 = r2_score(y_test_orig, y_pred_orig)

    return {"model": svr, "mse": mse, "r2": r2}
