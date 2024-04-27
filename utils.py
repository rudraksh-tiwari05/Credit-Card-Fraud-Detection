import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(data_path, target_var):
    df = pd.read_csv(data_path)
    X = df.drop(target_var, axis=1)
    y = df[target_var]
    return X, y

def encode_categorical(X):
    categorical_cols = [col for col in X.columns if X[col].dtype == "object"]
    for col in categorical_cols:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])
    return X

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def split_data(X, y, test_size, random_state):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
