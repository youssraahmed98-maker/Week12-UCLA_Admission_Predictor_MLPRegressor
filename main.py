from src.data.make_dataset import load_and_preprocess_data
from src.visualization.visualize import (
    plot_correlation_heatmap,
    plot_actual_vs_predicted
)
from src.features.build_features import build_features
from src.models.train_model import train_NNmodel
from src.models.predict_model import evaluate_model


if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "data/raw/Admission.csv"
    df = load_and_preprocess_data(data_path)

    # Plot correlation heatmap
    plot_correlation_heatmap(df)

    # Build features and target
    X, y = build_features(df)

    # Train the neural network model
    model, X_test_scaled, y_test = train_NNmodel(X, y)

    # Evaluate the model
    r2, mae, rmse = evaluate_model(model, X_test_scaled, y_test)

    # Predict again for visualization
    y_pred = model.predict(X_test_scaled)

    # Plot actual vs predicted
    plot_actual_vs_predicted(y_test, y_pred)

    # Print evaluation metrics
    print(f"R2 Score: {r2}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")