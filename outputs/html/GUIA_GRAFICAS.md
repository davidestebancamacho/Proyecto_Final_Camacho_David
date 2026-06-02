# 📊 Guía para Ver las Gráficas

## ¿Cómo ejecutar los scripts para generar todas las gráficas?

### Opción 1: Ejecutar TODO en orden (recomendado)

```bash
# 1. Ve a la carpeta del proyecto
cd /Users/davidesteban25/Desktop/Proyecto_Final_Camacho_David

# 2. Ejecuta los scripts en este orden:

# Primero: Análisis de Series de Tiempo (8 gráficas)
python3 src/seccion_series_tiempo.py

# Segundo: Análisis ML de popularidad (8 gráficas)
python3 src/proyecto_peliculas_ml.py

# Tercero: Visualizaciones complementarias (8 gráficas)
python3 src/visualizaciones_complementarias.py
```

**⏱️ Tiempo total: ~5-10 minutos**

---

## 📂 Dónde están las gráficas

Todas las gráficas se guardan en: `outputs/html/`

### Gráficas de Series de Tiempo (8 archivos)
- `viz_ts0_estacionariedad.html` - Análisis de estacionariedad
- `viz_ts1_serie_original.html` - Serie original de ingresos
- `viz_ts2_descomposicion_stl.html` - Descomposición tendencia + estacionalidad
- `viz_ts3_walkforward.html` - Validación walk-forward
- `viz_ts4_pronostico_4modelos.html` - Pronósticos de 4 modelos
- `viz_ts5_comparacion_metricas.html` - Comparación de métricas de error
- `viz_ts6_perfil_estacional.html` - Patrón estacional mensual
- `viz_ts7_ets_comparacion.html` - Mejor modelo (ETS)

### Gráficas de ML - Popularidad (8 archivos)
- `viz_1_distribucion_features.html` - Distribuciones de variables
- `viz_2_correlaciones.html` - Matriz de correlaciones
- `viz_3_boxplot_generos.html` - Popularidad por género
- `viz_4_radar_chart.html` - Perfil de características principales
- `viz_5_residuos.html` - Análisis de residuos del modelo
- `viz_6_shap_importancia.html` - Importancia de variables (SHAP)
- `viz_7_shap_beeswarm.html` - Impacto de cada variable
- `viz_8_shap_dependencia.html` - Dependencia parcial SHAP

### Gráficas Complementarias (8 archivos)
- `viz_c1_roi_scatter.html` - ROI por presupuesto
- `viz_c2_genero_año_heatmap.html` - Heatmap género × año
- `viz_c3_error_por_año.html` - Error del modelo por año
- `viz_c4_acf_pacf.html` - Autocorrelación de la serie
- `viz_c5_sarima_diagnostics.html` - Diagnósticos del modelo SARIMA
- `viz_c6_prophet_descomposicion.html` - Descomposición Prophet
- `viz_c7_sample_weights_violin.html` - Distribución de pesos
- `viz_c8_power_transformer.html` - Transformación de popularidad

---

## 🔍 Cómo abrir las gráficas

### Opción A: Desde VS Code
1. Abre VS Code
2. Ve a la carpeta `outputs/html/`
3. Haz clic derecho en cualquier `.html`
4. Selecciona "Open with Live Server" o "Open in Default Browser"

### Opción B: Desde la terminal
```bash
# Abre una gráfica específica en el navegador
open outputs/html/viz_ts1_serie_original.html

# O abre el navegador en la carpeta completa
open outputs/html/
```

---

## ✅ Estado actual

**Gráficas generadas hasta ahora:**
- ✅ Series de Tiempo: 8/8 (`viz_ts0` a `viz_ts7`)
- ⏳ ML: Ejecutándose...
- ⏳ Complementarias: Necesita que terminen las anteriores

---

## 💡 Qué hacen los scripts

| Script | Qué hace | Tiempo |
|--------|----------|--------|
| `seccion_series_tiempo.py` | Analiza ingresos mensuales con SARIMA, ETS, Prophet | 2-3 min |
| `proyecto_peliculas_ml.py` | Predice popularidad con XGBoost + SHAP | 3-5 min |
| `visualizaciones_complementarias.py` | Conecta ambos análisis con gráficas adicionales | 1-2 min |

---

## 🎯 Próximos pasos

1. Abre la terminal en VS Code
2. Ejecuta: `python3 src/proyecto_peliculas_ml.py`
3. Espera a que termine (verás mensajes)
4. Ejecuta: `python3 src/visualizaciones_complementarias.py`
5. Abre las gráficas en el navegador
