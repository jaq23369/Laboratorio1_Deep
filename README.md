# Laboratorio 1: Entrenamiento de Redes Neuronales (MLP)

**Curso:** CC3092 - Deep Learning y Sistemas Inteligentes
**Autor:** Joel Jaquez

## Objetivo

Entrenar un Perceptrón Multicapa (MLP) en PyTorch para resolver un problema de regresión: predecir el valor mediano de una vivienda (`median_house_value`) a partir de características socioeconómicas y geográficas de distritos censales de California, usando el dataset público California Housing Prices.

## Estructura del repositorio

```
Laboratorio1/
├── Data/
│   └── housing.csv          # Dataset original (Kaggle: camnugent/california-housing-prices)
├── notebooks/
│   └── lab1_mlp.ipynb       # Notebook completo: EDA, investigación, entrenamiento y evaluación
├── informe/
│   └── Laboratorio1_Deep.pdf # Informe escrito (investigación, tabla de resultados, discusión)
├── .gitignore
└── README.md
```

## Cómo correr el notebook

El proyecto usa un entorno virtual de Python (`.venv`), para no instalar las librerías de forma global.

```bash
cd Laboratorio1
python3 -m venv .venv
source .venv/bin/activate
pip install torch pandas numpy matplotlib scikit-learn jupyter ipykernel
jupyter notebook notebooks/lab1_mlp.ipynb
```

El notebook detecta automáticamente el dispositivo disponible (MPS en Apple Silicon, CUDA o CPU) y corre de principio a fin sin errores.

## Resumen del notebook, sección por sección

### 1. Dataset

Se usa el CSV de California Housing Prices: 20,640 observaciones (distritos censales) y 10 variables (9 features más el target `median_house_value`).

### 2. Exploración y preparación de los datos

Se responden las preguntas de exploración pedidas en el enunciado, y se documentan los siguientes hallazgos:

- **Nulos:** solo `total_bedrooms` tiene valores faltantes (207 de 20,640 filas, menos del 1%). Se imputan con la mediana, calculada únicamente sobre el conjunto de entrenamiento.
- **Duplicados:** no se encontró ninguna fila duplicada.
- **Outliers y capping:** `median_house_value` está truncado en \$500,001 (965 distritos, ~4.7%) y `housing_median_age` en 52 (1,273 distritos, ~6.2%). Se decide no eliminar estas filas, ya que son casos reales del dataset original, no errores.
- **Variables numéricas vs. categóricas:** 8 numéricas y 1 categórica (`ocean_proximity`, 5 categorías), codificada con one-hot encoding.
- **Escalado:** las variables numéricas (features y target) se escalan con `StandardScaler`, ajustado solo con el conjunto de entrenamiento.
- **División de datos:** 70% train / 15% validation / 15% test, respetando la regla de que el conjunto de test nunca influye en decisiones de entrenamiento ni de ajuste de hiperparámetros.

### 3. Investigación: capas de PyTorch y optimizadores

Se investigan y documentan, con ejemplos ejecutables y fuentes oficiales de PyTorch, los siguientes componentes:

- **Capas:** `nn.Linear`, funciones de activación (`nn.ReLU`, `nn.LeakyReLU`, `nn.Tanh`), `nn.Dropout`, `nn.BatchNorm1d`.
- **Funciones de pérdida:** `nn.MSELoss`, `nn.L1Loss`, `nn.SmoothL1Loss`.
- **Optimizadores:** `torch.optim.SGD`, `torch.optim.Adam`, `torch.optim.RMSprop`, incluyendo el rol de `lr` y `weight_decay`, y qué diferencia a cada optimizador de los demás.

### 4. Entrenamiento e iteración de hiperparámetros

Se define una clase `MLP` configurable y una función de entrenamiento reutilizable. Los hiperparámetros se exploran con **Random Search** (no grid search, dado que con 7 hiperparámetros un grid completo crecería exponencialmente): se muestrean 15 configuraciones al azar, variando simultáneamente arquitectura, activación, optimizador, `lr`, batch size, epochs y regularización (L1, L2, dropout o combinaciones), usando una semilla fija para que el muestreo sea reproducible.

Se grafican las curvas de pérdida de entrenamiento y validación de 3 iteraciones (mejor, mediana y peor según validación), lo que permite identificar señales de overfitting leve en la mejor configuración y de underfitting/inestabilidad en la peor.

### 5. Documentación de resultados

Tabla resumen de las 15 iteraciones (arquitectura, activación, optimizador/LR, batch/epochs, regularización, MSE, MAE y RMSE de validación), ordenada de mejor a peor desempeño.

### 6. Discusión y análisis

Esta sección se responde únicamente en el informe en PDF (`informe/Laboratorio1_Deep.pdf`).

## Resultado final

La mejor configuración encontrada (arquitectura `[32, 64, 16]`, ReLU, Adam con `lr≈0.0011`, batch de 128, 80 epochs, regularización L2 ligera) se evaluó una única vez sobre el conjunto de test:

| Métrica | Validación | Test |
|---|---:|---:|
| MSE | ≈2.67 x 10⁹ | ≈2.89 x 10⁹ |
| MAE | \$34,209 | \$35,905 |
| RMSE | \$51,715 | \$53,749 |

La cercanía entre las métricas de validación y de test indica que el modelo generaliza razonablemente bien, sin overfitting severo.
