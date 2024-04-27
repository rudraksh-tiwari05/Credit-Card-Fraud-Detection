from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model

def train_isolation_forest(X_train):
    model = IsolationForest(random_state=RANDOM_STATE)
    model.fit(X_train)
    return model

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model
