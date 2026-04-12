from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import pickle
import os


def train_NNmodel(X, y):
    # Create models folder
    os.makedirs("models", exist_ok=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    # Scale data (VERY important for neural networks)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Neural Network model
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=123
    )

    model.fit(X_train_scaled, y_train)

    # Save model
    with open("models/NNmodel.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save scaler
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return model, X_test_scaled, y_test