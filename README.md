# Predicción de Churn de Clientes

Modelo de clasificación binaria para predecir qué clientes de una empresa de telecomunicaciones tienen mayor riesgo de darse de baja (`Churn`), a partir de datos demográficos, de contrato y de servicios contratados.

## Estructura del proyecto

```
churn-project/
├── data/
│   ├── clientes.csv              # dataset original
│   └── predicciones.csv          # generado al ejecutar predict.py
├── notebooks/                    # documentación del proceso de análisis
│   ├── 01_eda.ipynb
│   ├── 02_preprocesado.ipynb
│   └── 03_modelado.ipynb
├── src/                          # código reutilizable
│   ├── preprocessing.py          # limpieza + encoding + escalado
│   ├── train.py                  # entrena y guarda el modelo
│   └── predict.py                # predice sobre clientes nuevos
├── models/                       # generado al ejecutar train.py
│   ├── model.joblib
│   └── preprocessor.joblib
├── requirements.txt
└── README.md
```

## Instalación

```
pip install -r requirements.txt
```

## Cómo ejecutar

### 1. Explorar el análisis (opcional)

Los notebooks en `notebooks/` documentan el proceso completo: EDA → preprocesado → modelado, con las decisiones justificadas paso a paso. Ábrelos en VS Code con la extensión Jupyter.

### 2. Entrenar el modelo

```
cd src
python train.py
```

Esto carga `data/clientes.csv`, separa train/test, entrena un Random Forest y guarda el modelo y el preprocesador en `models/`. Imprime las métricas de evaluación por consola.

### 3. Predecir clientes nuevos

```
cd src
python predict.py ../data/clientes.csv --output ../data/predicciones.csv
```

El CSV de entrada debe tener las mismas columnas que `clientes.csv` (si incluye la columna `Churn`, se ignora automáticamente — es justo lo que se predice). El resultado incluye, por cada cliente, la predicción (`Yes`/`No`) y la probabilidad de churn.

Para probarlo con clientes nuevos de verdad, sustituye la ruta por tu propio CSV con el mismo formato (mismas columnas que `clientes.csv`, sin `Churn`).

También se puede usar como función dentro de otro script de Python:
```python
from src.predict import predecir
resultado = predecir(df_nuevos_clientes)
```

## Metodología (resumen)

- **Preprocesado**: limpieza de `TotalCharges` (texto → numérico), simplificación de categorías redundantes, codificación binaria y Label Encoding para variables multicategoría, escalado de variables numéricas. El ajuste (`fit`) del encoder/scaler se hace solo con datos de entrenamiento, para evitar fuga de información hacia el conjunto de test.
- **Modelos comparados**: Árbol de Decisión (baseline), Random Forest y XGBoost. Se eligieron modelos basados en árboles por el tipo de variables (mayoritariamente categóricas), la probable existencia de interacciones entre variables, y por dar feature importance interpretable para el negocio.
- **Métrica principal**: F1-score, elegido por el desbalanceo de clases (~27% Churn / 73% No Churn), ya que equilibra precision y recall mejor que accuracy en este contexto.
- **Modelo final**: Random Forest (mejor F1-score de los tres).

## Conclusiones

**Qué funciona:** el modelo distingue razonablemente bien entre clientes que se quedan y que se van (ROC-AUC ≈ 0.82). Las variables más determinantes son `MonthlyCharges`, `TotalCharges`, `tenure` y `Contract` — coherente con la intuición de negocio (clientes nuevos, con contratos cortos y cargos altos, son más propensos a irse).

**Limitaciones principales:**
- Recall de la clase Churn ≈ 50%: de cada 2 clientes que realmente se van a dar de baja, el modelo solo detecta 1. En un caso de uso real de retención, esto es un problema serio si el coste de "perder un cliente sin intentar retenerlo" es alto.
- El desbalanceo de clases no se corrigió explícitamente (no se usó `class_weight`, SMOTE, ni ajuste de umbral) — se dejó como línea de trabajo futura.
- F1-score trata precision y recall con el mismo peso, pero no está claro que ese sea el equilibrio correcto para el negocio sin conocer el coste real de cada tipo de error.
- Los modelos se entrenaron con hiperparámetros por defecto, sin búsqueda (`GridSearchCV`/`RandomizedSearchCV`).
- Evaluación con una sola partición train/test (80/20), no con validación cruzada.

**Próximos pasos, con más tiempo:**
1. Balancear la desigualdad Churn/no Churn, penalizar al modelo cuando falle un churn mas furte que cuando falle un no churn. 
2. Dejar escapar a un cliente que se iba cuesta mas que retener a alguien que se quedaba.Encontrar punto de equilibrio entre precision y recall segun el coste de cada error, no 50-50 como hace F1-Score
3. Feature Enguineering: Crear variables nuevas que combinen las existentes. EN EDA vimos que la relacion con churn parece depender de combinaciones, por ejemplo TotalCharges / Tenure --> gasto medio mensual.
4. Feature Importance: Sabemos que variables importan mas pero no sabemos como afectan (Si hay mas tenure, hay mas o menos churn?) 