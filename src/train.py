import sys
import os

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report,
)

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import ChurnPreprocessor

RANDOM_STATE = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'clientes.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def main():
    df = pd.read_csv(DATA_PATH)

    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df['Churn']
    )

    preprocessor = ChurnPreprocessor()
    X_train, y_train = preprocessor.fit_transform(df_train)
    X_test, y_test = preprocessor.transform(df_test)

    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print('Accuracy: ', round(accuracy_score(y_test, y_pred), 4))
    print('Precision:', round(precision_score(y_test, y_pred), 4))
    print('Recall:   ', round(recall_score(y_test, y_pred), 4))
    print('F1-score: ', round(f1_score(y_test, y_pred), 4))
    print('ROC-AUC:  ', round(roc_auc_score(y_test, y_proba), 4))
    print()
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, 'model.joblib'))
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, 'preprocessor.joblib'))
    print(f'Modelo y preprocesador guardados en {MODELS_DIR}')


if __name__ == '__main__':
    main()