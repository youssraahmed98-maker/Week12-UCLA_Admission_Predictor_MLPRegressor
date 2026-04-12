from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np


def evaluate_model(model, X_test_scaled, y_test):
    # Predict on the testing set
    y_pred = model.predict(X_test_scaled)

    # Calculate evaluation metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return r2, mae, rmse