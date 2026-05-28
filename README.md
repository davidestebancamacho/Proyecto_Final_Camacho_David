# 🎬 TMDB Movies ML — Predicción de Popularidad

Análisis completo de Machine Learning sobre el dataset TMDB Movies Metadata.  
**Pregunta central:** ¿Qué características de una película predicen mejor su popularidad?

---

## Contenido del proyecto

| Sección | Archivo | Qué cubre |
|---------|---------|-----------|
| 1–4 | `src/proyecto_peliculas_ml.py` | Exploración · Limpieza · Modelos · SHAP |
| 5 | `src/seccion_series_tiempo.py` | Análisis temporal · STL · SARIMA · Prophet |
| 6 | `src/visualizaciones_complementarias.py` | 8 visualizaciones adicionales |
| App | `src/app_gradio.py` | Mini-app interactiva con Gradio |

---

## Visualizaciones incluidas (22 en total)

**Proyecto principal**
- Distribuciones de variables clave
- Matriz de correlaciones interactiva
- Popularidad por género (boxplot)
- Comparación de modelos (radar 5 métricas)
- R² por segmento de popularidad
- Real vs Predicho + residuos
- SHAP: Bar · Beeswarm · Dependence Plot

**Series de tiempo**
- Serie temporal con media móvil
- Descomposición STL (4 componentes)
- Walk-forward validation visual
- Pronóstico Real vs 3 modelos
- Comparación de métricas de pronóstico
- Perfil estacional mensual

**Complementarias**
- ROI scatter: Budget vs Revenue
- Heatmap evolución por género × año
- Error del modelo por año (deriva temporal)
- ACF / PACF interactivo (justifica SARIMA)
- Diagnóstico de residuos SARIMA
- Prophet descompuesto (trend + yearly)
- Sample weights por quintil (balanceo)
- PowerTransformer vs log1p vs raw

---

## Decisiones técnicas clave

### Predicción (Sección 2)
- **Sin data leakage**: imputación, log1p y StandardScaler dentro del Pipeline
- **Stratified split**: garantiza representación uniforme de películas virales
- **Hiperparámetros**: RidgeCV automático · RandomizedSearchCV para RF y XGBoost
- **Balanceo**: PowerTransformer(yeo-johnson) + sample_weights inversamente proporcionales
- **Smoke tests**: 10 verificaciones antes del entrenamiento completo

### Explicabilidad (Sección 3)
- SHAP TreeExplainer sobre el mejor modelo
- Visualizaciones en Plotly (interactivas, no estáticas)

### Series de tiempo (Sección 5)
- Variable: revenue medio mensual 1990–2017 (no popularity, que es un snapshot)
- Validación: walk-forward con expanding window · h=12 meses
- Métricas: MAE · RMSE · MAPE · SMAPE (no R²)
- Modelos: Naive estacional · SARIMA (orden por AIC) · Prophet

---

## Instalación

```bash
# 1. Clonar el repo
git clone https://github.com/TU_USUARIO/tmdb-movies-ml.git
cd tmdb-movies-ml

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Mac / Linux
# venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar el dataset
# Ver instrucciones en data/README.md
```

---

## Ejecución

### En orden (notebook / Colab)
```bash
# Ejecutar en secuencia — cada archivo asume el anterior en memoria
python src/proyecto_peliculas_ml.py
python src/seccion_series_tiempo.py
python src/visualizaciones_complementarias.py
```

### App interactiva
```bash
python src/app_gradio.py
# Abre http://localhost:7860
```

### Exportar visualizaciones a HTML
Descomenta `save(fig, "nombre")` en cualquier archivo — genera HTMLs
standalone en `outputs/html/` que se abren en el browser sin Python.

---

## Estructura del repositorio

```
tmdb-movies-ml/
├── src/
│   ├── proyecto_peliculas_ml.py          # Análisis principal
│   ├── seccion_series_tiempo.py          # Series de tiempo
│   ├── visualizaciones_complementarias.py # 8 vizs adicionales
│   └── app_gradio.py                     # Mini-app
├── notebooks/                            # Versiones .ipynb (opcional)
├── outputs/
│   └── html/                             # Gráficos exportados
├── data/
│   └── README.md                         # Instrucciones de descarga
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Resultados principales

| Modelo | R² test | MAE | MAPE |
|--------|---------|-----|------|
| Ridge (RidgeCV) | ~0.51 | — | — |
| Random Forest | ~0.71 | — | — |
| **XGBoost** | **~0.74** | — | — |

> Los valores exactos dependen del split aleatorio. Ejecutar el código para ver los números actualizados.

**Variables más importantes (SHAP):** `log_votes` › `log_revenue` › `log_budget` › `vote_average`

---

## Stack tecnológico

`Python 3.10+` · `pandas` · `numpy` · `scikit-learn` · `xgboost` · `shap`  
`statsmodels` · `prophet` · `plotly` · `gradio` · `scipy`

---

## Uso de IA en el desarrollo

Este proyecto fue desarrollado con asistencia de **Claude (Anthropic)** en todas las etapas:
exploración, limpieza, modelado, explicabilidad SHAP, análisis de series de tiempo y mini-app.
Ver Sección 5 del código principal para la reflexión completa sobre el uso de IA.
