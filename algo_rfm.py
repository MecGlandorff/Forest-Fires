



from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


def random_forest_model(data):
    # Separate features and target pred value
    X = data.drop(columns=['area'])
    y = data['area']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Regressor
    rf = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=None)
    rf.fit(X_train, y_train)

    # Predict
    y_pred = rf.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {"model": rf, "mse": mse, "r2": r2}


def tune_random_forest(data):
    X = data.drop(columns=['area'])
    y = data['area']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Gridsearchcv
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='r2', verbose=2)
    grid_search.fit(X_train, y_train)

    # Best model
    best_rf = grid_search.best_estimator_

    # Predict
    y_pred = best_rf.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Best Parameters: {grid_search.best_params_}")
    return {"model": best_rf, "mse": mse, "r2": r2}