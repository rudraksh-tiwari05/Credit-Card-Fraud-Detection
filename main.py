import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import load_data, encode_categorical, scale_features, split_data
from models import train_random_forest, train_isolation_forest, train_logistic_regression

# Load data
X, y = load_data(DATA_PATH, TARGET_VAR)

# Preprocess data
X = encode_categorical(X)
X = scale_features(X)

# Split data
X_train, X_test, y_train, y_test = split_data(X, y, TEST_SIZE, RANDOM_STATE)

# Train models
models = {
    "RandomForest": train_random_forest(X_train, y_train),
    "IsolationForest": train_isolation_forest(X_train),
    "LogisticRegression": train_logistic_regression(X_train, y_train),
}

# Evaluate models
for model_name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

