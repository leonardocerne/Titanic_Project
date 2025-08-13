from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_logistic_regression(X_train, y_train, random_state=42):
    print("Treinando modelo de regressão logística..")
    model = LogisticRegression(random_state=random_state, solver='liblinear', max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, random_state=42):
    print("Treinando modelo de Floresta Aleatória...")
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    print("Floresta Aleatória treinada.")
    return model