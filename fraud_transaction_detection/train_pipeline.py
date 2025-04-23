from sklearn.metrics import (
    classification_report,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from fraud_transaction_detection.pipeline import fraud_detection_pipeline
from fraud_transaction_detection.processing.data_manager import (
    get_training_data,
    save_pipeline,
)


def run_training() -> None:
    X_train, X_test, y_train, y_test = X_train, X_test, y_train, y_test = (
        get_training_data()
    )

    # Pipeline fitting
    fraud_detection_pipeline.fit(X_train, y_train)
    print("Pipeline fitting done")
    print(
        "---------------`-------`--------------------prediction starts---------------`-------`--------------------"
    )
    save_pipeline(pipeline_to_persist=fraud_detection_pipeline)

    y_pred = fraud_detection_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    y_scores = fraud_detection_pipeline.predict_proba(X_test)[
        :, 1
    ]  # Probability for class 1 (fraud)
    roc_auc = roc_auc_score(y_test, y_scores)
    print(f"ROC AUC Score: {roc_auc:.4f}")


if __name__ == "__main__":
    run_training()
