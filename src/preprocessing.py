import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class ChurnPreprocessor:
    """Limpia, codifica y escala los datos crudos de clientes de telco."""

    COLS_SIMPLIFICAR_INTERNET = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
    ]

    COLS_BINARIAS = [
        'gender', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'PaperlessBilling',
    ]

    COLS_MULTICATEGORIA = ['Contract', 'PaymentMethod', 'InternetService']

    COLS_NUMERICAS = ['tenure', 'MonthlyCharges', 'TotalCharges']

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns_ = None
        self.is_fitted_ = False

    def _limpiar(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'])

        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(0)

        for col in self.COLS_SIMPLIFICAR_INTERNET:
            df[col] = df[col].replace('No internet service', 'No')
        df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')

        for col in self.COLS_BINARIAS:
            df[col] = df[col].map(lambda x: 1 if x in ['Yes', 'Male'] else 0)

        return df

    def _extraer_target(self, df: pd.DataFrame):
        """Separa y codifica la columna Churn si está presente."""
        if 'Churn' in df.columns:
            y = df['Churn'].map({'Yes': 1, 'No': 0})
            df = df.drop(columns=['Churn'])
            return df, y
        return df, None

    def fit(self, df: pd.DataFrame):
        """Ajusta encoders y scaler solo con datos de entrenamiento."""
        df = self._limpiar(df)
        df, _ = self._extraer_target(df)

        for col in self.COLS_MULTICATEGORIA:
            le = LabelEncoder()
            le.fit(df[col])
            self.label_encoders[col] = le

        for col in self.COLS_MULTICATEGORIA:
            df[col] = self.label_encoders[col].transform(df[col])

        self.scaler.fit(df[self.COLS_NUMERICAS])

        self.feature_columns_ = df.columns.tolist()
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame):
        """Aplica la limpieza + encoders + scaler ya ajustados."""
        if not self.is_fitted_:
            raise RuntimeError("El preprocesador no está ajustado. Llama a fit() primero.")

        df = self._limpiar(df)
        df, y = self._extraer_target(df)

        for col in self.COLS_MULTICATEGORIA:
            le = self.label_encoders[col]
            df[col] = df[col].map(
                lambda val: le.transform([val])[0] if val in le.classes_ else -1
            )

        df[self.COLS_NUMERICAS] = self.scaler.transform(df[self.COLS_NUMERICAS])
        df = df[self.feature_columns_]

        if y is not None:
            return df, y
        return df

    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        return self.transform(df)