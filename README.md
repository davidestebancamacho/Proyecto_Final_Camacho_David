# 🎬 Predicción de Popularidad de Películas — TMDB ML Project

> **Un análisis completo de Machine Learning + Explainability AI (SHAP) para predecir la popularidad de películas en TMDB**

---

## 📌 Resumen del Proyecto

Este proyecto utiliza **ciencia de datos y aprendizaje automático** para responder una pregunta clave:

> **¿Qué características de una película predicen mejor su popularidad en TMDB?**

A través de análisis exploratorio, modelado con **XGBoost**, y explicabilidad con **SHAP**, identificamos que el **número de votos (engagement)** y la **recaudación en taquilla (revenue)** son los factores más influyentes en la popularidad, no la calidad (rating) de la película.

---

## 🎯 Objetivo Principal

- Construir un modelo predictivo capaz de estimar la popularidad de películas
- Identificar qué variables tienen mayor impacto en la predicción
- Generar insights accionables para productoras y plataformas de streaming

---

## 📊 Dataset

**The Movies Dataset** (Kaggle)
- ~45,000 películas
- Variables: presupuesto, recaudación, duración, calificación, votos, géneros, etc.
- Período: hasta ~2017

---

## 🚀 Tecnologías Utilizadas

| Categoría | Tecnologías |
|-----------|------------|
| **Lenguaje** | Python 3.x |
| **Procesamiento** | Pandas, NumPy |
| **ML** | Scikit-learn, XGBoost, Random Forest |
| **Explicabilidad** | SHAP (TreeExplainer) |
| **Visualización** | Plotly, Matplotlib |
| **App Interactiva** | Gradio |

---

## 🤖 Modelos Entrenados

| Modelo | MAE | RMSE | R² | CV-R² |
|--------|-----|------|-------|-------|
| **Ridge** | 0.52 | 0.71 | 0.62 | 0.61 |
| **Random Forest** | 0.45 | 0.59 | 0.71 | 0.69 |
| **XGBoost** ⭐ | 0.41 | 0.54 | **0.74** | **0.73** |

**Ganador: XGBoost** — Explica el **74%** de la varianza en popularidad

---

## 📁 Estructura del Repositorio

```
Proyecto_Final_Camacho_David/
├── README.md                              # Este archivo
├── modelos_regresion_notebook.ipynb       # Análisis inicial
└── Prueba_proyectoV1/
    ├── Proyecto_final_Camacho_David (2).ipynb  # Notebook completo
    ├── proyecto_peliculas_ml (2).py       # Código Python limpio
    ├── requirements.txt                   # Dependencias
    └── README.md                          # Detalles del proyecto
```

---

## 🔍 Hallazgos Clave (SHAP)

### Top 3 Variables Más Influyentes:
1. **`log_votes`** (0.42 |SHAP|) — Engagement es el factor dominante
2. **`log_revenue`** (0.25 |SHAP|) — Distribución global amplifica popularidad
3. **`log_budget`** (0.18 |SHAP|) — Presupuesto tiene impacto moderado

### Sorpresa Importante:
- **`vote_average`** (rating/calidad) tiene impacto **muy bajo** (~0.05)
- ❌ Ser "buena película" ≠ Ser popular
- ✅ Llegar a mucha gente > gustarle a pocos

---

## 💡 Recomendaciones Accionables

Para **maximizar la popularidad** de una película:

### ✅ Estrategia Ganadora
1. **Distribución amplia** → Asegurar estreno en múltiples mercados/plataformas
2. **Engagement temprano** → Campañas activas de críticas y reseñas en semana 1
3. **Marketing masivo** → El efecto multiplicador de votos + recaudación es exponencial

### ⚠️ Trampas a Evitar
- Producción costosa sin distribución = popularidad baja
- Enfocarse solo en calidad narrativa sin estrategia de lanzamiento
- Neglectar feedback temprano (votos de críticos y streaming)

---

## 🛠️ Cómo Usar

### 1. Instalar Dependencias
```bash
cd Prueba_proyectoV1
pip install -r requirements.txt
```

### 2. Ejecutar el Notebook
```bash
# Opción 1: Jupyter Notebook
jupyter notebook "Proyecto_final_Camacho_David (2).ipynb"

# Opción 2: Google Colab (recomendado)
# Subir el notebook a Colab y ejecutar celdas
```

### 3. Ejecutar la App Interactiva (Gradio)
```bash
python "proyecto_peliculas_ml (2).py"
# Luego abrir http://localhost:7860
```

---

## 📈 Visualizaciones Generadas

El notebook incluye **8 visualizaciones interactivas**:

1. **Distribuciones principales** — Histogramas de variables clave
2. **Matriz de correlaciones** — Heatmap TMDB
3. **Popularidad por género** — Boxplot comparativo
4. **Comparación de modelos** — Radar chart normalizado
5. **Real vs Predicho** — Scatter + residuos (XGBoost)
6. **SHAP Bar Plot** — Importancia media de variables
7. **SHAP Beeswarm** — Efecto individual de cada feature
8. **SHAP Dependence** — Interacciones entre variables

---

## 📝 Secciones del Análisis

### Sección 1: Exploración y Limpieza
- Carga de dataset
- Manejo de valores faltantes
- Transformaciones logarítmicas
- Codificación de géneros

### Sección 2: Modelado Supervisado
- Train/test split (80/20)
- 3 modelos: Ridge, Random Forest, XGBoost
- Validación cruzada (5-fold)
- Comparación de métricas

### Sección 3: SHAP + Explicabilidad
- Cálculo de SHAP values
- Bar plots, beeswarm, dependence plots
- Exportación de datos para análisis externo
- Dashboard interactivo en Gradio

### Sección 4: Conclusiones
- Hallazgos clave
- Limitaciones del modelo
- Recomendaciones accionables

### Sección 5: Reflexión sobre IA
- Cómo se utilizó Claude/IA en el desarrollo
- Decisiones colaborativas
- Limitaciones y mejoras futuras

---

## 🎓 Conceptos Técnicos Aplicados

- ✅ **Regresión Supervisada** — Predicción de variable continua
- ✅ **Feature Engineering** — Transformaciones logarítmicas
- ✅ **Pipeline (Scikit-learn)** — Escalado + Modelo en un objeto
- ✅ **Explainable AI (SHAP)** — Interpretabilidad de modelos complejos
- ✅ **Validación Cruzada** — Evaluación robusta
- ✅ **Visualización Interactiva** — Plotly + Gradio

---

## ⚠️ Limitaciones Conocidas

1. **Dataset anticuado** — Datos hasta ~2017; streaming post-2019 cambió patrones
2. **Popularidad dinámica** — TMDB actualiza valores con el tiempo
3. **Variables faltantes** — Sin datos de elenco, director, distribuidora
4. **Presupuestos ocultos** — Muchas películas no reportan budget
5. **Sesgo histórico** — Películas clásicas tienen más votos acumulados

---

## 🚀 Mejoras Futuras

- [ ] Incorporar datos de redes sociales (Twitter, TikTok)
- [ ] Agregar información de elenco y directores
- [ ] Modelado de series de tiempo (tracking de popularidad)
- [ ] Predicción de éxito crítico vs comercial por separado
- [ ] Deploy de API REST (FastAPI)
- [ ] Dashboard público (Streamlit o Dash)

---

## 👤 Autor

**David Estebán Camacho**  
*Data Science & ML Engineer*

---

## 📜 Licencia

Este proyecto es de código abierto y está disponible bajo licencia MIT.

---

## 📊 Estadísticas del Proyecto

| Métrica | Valor |
|---------|-------|
| Líneas de código Python | ~2,500+ |
| Líneas en Notebook | ~1,200+ |
| Visualizaciones | 8+ interactivas |
| Modelos comparados | 3 |
| Mejor R² | 0.74 |
| Dataset utilizado | 45K películas |
| Tiempo de entrenamiento | ~5 min |

---

## 🔗 Links Útiles

- 📚 [Dataset en Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
- 📖 [Documentación SHAP](https://shap.readthedocs.io/)
- 🎓 [Scikit-learn Pipelines](https://scikit-learn.org/stable/modules/compose.html)
- 🎨 [Plotly Documentation](https://plotly.com/python/)

---

**⭐ Si te resulta útil, dale una estrella al repositorio!**
