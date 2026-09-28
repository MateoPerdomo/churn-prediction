import sys
import os
import argparse

import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import ChurnPreprocessor  # noqa: F401 (necesario para joblib.load)

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def cargar_artefactos():
    model = joblib.load(os.path.join(MODELS_DIR, 'model.joblib'))
    preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.joblib'))
    return model, preprocessor


def predecir(df_nuevos: pd.DataFrame, model=None, preprocessor=None) -> pd.DataFrame:
    if model is None or preprocessor is None:
        model, preprocessor = cargar_artefactos()

    df_features = df_nuevos.drop(columns=['Churn'], errors='ignore')
    X = preprocessor.transform(df_features)

    predicciones = model.predict(X)
    probabilidades = model.predict_proba(X)[:, 1]

    resultado = pd.DataFrame({
        'Churn_predicho': ['Yes' if p == 1 else 'No' for p in predicciones],
        'Probabilidad_Churn': probabilidades.round(4),
    })

    if 'customerID' in df_nuevos.columns:
        resultado.insert(0, 'customerID', df_nuevos['customerID'].values)

    return resultado


def main():
    parser = argparse.ArgumentParser(
        description='Predice churn para uno o varios clientes nuevos, dados en un CSV crudo.'
    )
    parser.add_argument('input_csv', help='Ruta al CSV con clientes nuevos (mismo formato que clientes.csv, sin columna Churn)')
    parser.add_argument('--output', default=None, help='Ruta donde guardar el CSV con las predicciones')
    args = parser.parse_args()

    df_nuevos = pd.read_csv(args.input_csv)
    resultado = predecir(df_nuevos)

    print(resultado.to_string(index=False))

    if args.output:
        resultado.to_csv(args.output, index=False)
        print(f'\nResultado guardado en {args.output}')


if __name__ == '__main__':
    main()
