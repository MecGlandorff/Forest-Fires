# In algo.py
def svr(X, y):
    """SVR model training and evaluation."""

    # Split 80/20, train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale X and y
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

    # Train SVR
    svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr_model.fit(X_train, y_train)

    # Predict
    y_pred = svr_model.predict(X_test)

    # Inverse transform predictions
    y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Evaluate
    mse = mean_squared_error(y_test_orig, y_pred_orig)
    r2 = r2_score(y_test_orig, y_pred_orig)

    return {"model": svr_model, "mse": mse, "r2": r2}
