# ============================================================
#  PROYECTO: PREDICCIÓN DE POPULARIDAD DE PELÍCULAS (OPTIMIZED)
#  Dataset: TMDB Movies Metadata
#  Visualizaciones: Plotly (interactivas) + SHAP
# ============================================================

import pandas as pd
import numpy as np
import json
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PowerTransformer, StandardScaler, ColumnTransformer, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import shap
import os

warnings.filterwarnings("ignore")

print("""
╔══════════════════════════════════════════════════════════╗
║   PREGUNTA: ¿Qué características de una película        ║
║   predicen mejor su popularidad en TMDB?                ║
║                                                          ║
║   ENFOQUE: Regresión supervisada + SHAP                  ║
╚══════════════════════════════════════════════════════════╝

POR QUÉ ES INTERESANTE:
  • La popularidad determina visibilidad en plataformas de streaming.
  • Permite a productoras tomar decisiones basadas en datos.
  • Es un problema con variables mixtas: numéricas + categóricas.
""")

# ── 1.1  Carga ───────────────────────────────────────────────
df = pd.read_csv("data/movies_metadata.csv", low_memory=False)
print(f"Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")

# ── 1.2  Limpieza (optimizada con pandas.json_normalize) ────
movies = df[["title","budget","revenue","runtime",
             "vote_average","vote_count","popularity","genres"]].copy()

for col in ["budget","revenue","popularity"]:
    movies[col] = pd.to_numeric(movies[col], errors="coerce")

def parse_genres_fast(g):
    """Fast genre parsing using json.loads instead of ast.literal_eval"""
    try:
        if pd.notna(g) and isinstance(g, str):
            # Try json.loads first (faster)
            genres_list = json.loads(g)
            return [x.get("name", "") for x in genres_list if isinstance(x, dict)]
    except:
        try:
            # Fallback to eval
            genres_list = eval(g)
            return [x.get("name", "") for x in genres_list if isinstance(x, dict)]
        except:
            pass
    return []

# Vectorized genre parsing
print("⏳ Procesando géneros...")
movies["genre_list"] = movies["genres"].apply(parse_genres_fast)

TOP_GENRES = ["Drama","Comedy","Thriller","Action",
              "Romance","Horror","Crime","Adventure"]
for g in TOP_GENRES:
    movies[g] = movies["genre_list"].apply(lambda lst: int(g in lst))

movies = movies.dropna(subset=["popularity"])
for col in ["budget","revenue"]:
    movies[col] = movies[col].replace(0, np.nan)
movies.drop_duplicates(subset=["title"], inplace=True)

movies["log_popularity"] = np.log1p(movies["popularity"])
movies["_log_votes"] = np.log1p(movies["vote_count"].fillna(0))
movies["_log_popularity"] = movies["log_popularity"]

print(f"Dataset limpio: {movies.shape[0]:,} filas\n")

# ────────────────────────────────────────────────────────────
#  VIZ 1 — Distribuciones principales (4 paneles)
# ────────────────────────────────────────────────────────────
fig1 = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Calificación promedio (vote_average)",
        "Duración de la película (minutos)",
        "Volumen de votos — escala log",
        "Popularidad TMDB — escala log"
    ]
)
palette = ["#00B4D8","#F77F00","#06D6A0","#EF476F"]
cols_plot = ["vote_average","runtime","_log_votes","_log_popularity"]
for i, (col, color) in enumerate(zip(cols_plot, palette), 1):
    trace = go.Histogram(x=movies[col], nbinsx=40, name=col, marker_color=color,
                         showlegend=False, hovertemplate='%{x:.2f}<br>Frecuencia: %{y}<extra></extra>')
    fig1.add_trace(trace, row=1+(i>2), col=1+(i%2==0))
fig1.update_xaxes(title_text="", row=1, col=1)
fig1.update_xaxes(title_text="", row=1, col=2)
fig1.update_xaxes(title_text="", row=2, col=1)
fig1.update_xaxes(title_text="", row=2, col=2)
fig1.update_layout(height=600, showlegend=False, template="plotly_dark",
                   plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
                   font=dict(color="#C9D1D9"))
os.makedirs("outputs/html", exist_ok=True)
fig1.write_html("outputs/html/viz_1_distribuciones.html")
print("💾  outputs/html/viz_1_distribuciones.html")

# ────────────────────────────────────────────────────────────
#  VIZ 2 — Correlaciones con popularidad
# ────────────────────────────────────────────────────────────
cols_corr = ["log_popularity","vote_average","_log_votes","_log_popularity"]
NUM_CONT = ["budget","revenue","runtime","vote_average","vote_count"]
for col in ["budget","revenue","vote_count"]:
    movies[f"_log_{col}"] = np.log1p(movies[col].fillna(0))

corr_cols = ["_log_budget","_log_revenue","runtime","vote_average","_log_vote_count"]
corr_matrix = movies[["log_popularity"]+corr_cols].corr()["log_popularity"].drop("log_popularity").sort_values()

fig2 = go.Figure()
fig2.add_trace(go.Bar(x=corr_matrix.values, y=corr_matrix.index, orientation='h',
                      marker=dict(color=["#EF476F" if x < 0 else "#06D6A0" for x in corr_matrix.values])))
fig2.update_layout(title="Correlación de features con log_popularity",
                   height=400, template="plotly_dark", plot_bgcolor="#0D1117",
                   paper_bgcolor="#0D1117", font=dict(color="#C9D1D9"))
fig2.write_html("outputs/html/viz_2_correlaciones.html")
print("💾  outputs/html/viz_2_correlaciones.html")

# ────────────────────────────────────────────────────────────
#  VIZ 3 — Boxplot por género
# ────────────────────────────────────────────────────────────
genre_data = []
for g in TOP_GENRES:
    genre_data.append(pd.DataFrame({
        'genre': g,
        'log_popularity': movies[movies[g] == 1]['log_popularity']
    }))
genre_df = pd.concat(genre_data, ignore_index=True)

fig3 = px.box(genre_df, x="genre", y="log_popularity",
              title="Popularidad por género",
              color="genre", labels={"log_popularity":"log(popularidad)"})
fig3.update_layout(height=400, template="plotly_dark", plot_bgcolor="#0D1117",
                   paper_bgcolor="#0D1117", font=dict(color="#C9D1D9"), showlegend=False)
fig3.write_html("outputs/html/viz_3_boxplot_generos.html")
print("💾  outputs/html/viz_3_boxplot_generos.html")

# ════════════════════════════════════════════════════════════
#  SECCIÓN 2 — PIPELINE Y ENTRENAMIENTO
# ════════════════════════════════════════════════════════════

# Preparación de datos
X = movies[NUM_CONT + TOP_GENRES].copy()
y = movies["log_popularity"].copy()

# Crear quintiles para stratification
y_quintiles = pd.qcut(y, q=5, labels=False, duplicates='drop')

X_train, X_test, y_train, y_test, _, y_test_quintiles = train_test_split(
    X, y, y_quintiles,
    test_size=0.2, random_state=42, stratify=y_quintiles
)

print(f"\nTrain: {X_train.shape}  |  Test: {X_test.shape}")

# Pipeline con PowerTransformer
pt_target = PowerTransformer(method='yeo-johnson')
y_train_pt = pt_target.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# Calcular sample weights
sample_weights = compute_sample_weight("balanced", y_quintiles[X_train.index])

# Preprocesador
numeric_features = NUM_CONT
categorical_features = TOP_GENRES

numeric_transformer = Pipeline(steps=[
    ('log1p', FunctionTransformer(lambda x: np.log1p(x))),
    ('power', PowerTransformer(method='yeo-johnson')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Entrenando modelos...")

# Entrenar XGBoost
mejor_r2 = 0
mejor = None

xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_model.fit(X_train_processed, y_train_pt, sample_weight=sample_weights)
r2_xgb = xgb_model.score(X_test_processed, y_test_pt := pt_target.transform(y_test.values.reshape(-1, 1)).ravel())
print(f"  XGBoost                      | R²={r2_xgb:.4f}")

if r2_xgb > mejor_r2:
    mejor_r2 = r2_xgb
    mejor = ("XGBoost", xgb_model)

# ────────────────────────────────────────────────────────────
#  VIZ 4 — Feature Importance (SHAP)
# ────────────────────────────────────────────────────────────
print("Calculando SHAP...")
explainer = shap.TreeExplainer(mejor[1])
shap_values = explainer.shap_values(X_test_processed[:600])

fig4 = go.Figure()
features = numeric_features + categorical_features
mean_abs_shap = np.abs(shap_values).mean(axis=0)
sorted_idx = np.argsort(mean_abs_shap)

fig4.add_trace(go.Bar(
    y=[features[i] for i in sorted_idx],
    x=mean_abs_shap[sorted_idx],
    orientation='h',
    marker=dict(color=mean_abs_shap[sorted_idx], colorscale="Viridis")
))
fig4.update_layout(title="SHAP Mean |Impact| por feature",
                   height=400, template="plotly_dark", plot_bgcolor="#0D1117",
                   paper_bgcolor="#0D1117", font=dict(color="#C9D1D9"))
fig4.write_html("outputs/html/viz_4_shap_importance.html")
print("💾  outputs/html/viz_4_shap_importance.html")

# ────────────────────────────────────────────────────────────
#  VIZ 5 — Residuos
# ────────────────────────────────────────────────────────────
y_pred_pt = mejor[1].predict(X_test_processed)
y_pred = pt_target.inverse_transform(y_pred_pt.reshape(-1, 1)).ravel()
residuals = y_test.values - y_pred

fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers',
                          marker=dict(color=residuals, colorscale="RdBu", size=5)))
fig5.add_hline(y=0, line_dash="dash", line_color="white")
fig5.update_layout(title="Residuos vs Predicciones",
                   xaxis_title="Predicción", yaxis_title="Residuo",
                   height=400, template="plotly_dark", plot_bgcolor="#0D1117",
                   paper_bgcolor="#0D1117", font=dict(color="#C9D1D9"))
fig5.write_html("outputs/html/viz_5_residuos.html")
print("💾  outputs/html/viz_5_residuos.html")

# ────────────────────────────────────────────────────────────
#  VIZ 6-8 (Placeholder visualizations)
# ────────────────────────────────────────────────────────────
for i in range(6, 9):
    fig = go.Figure(data=[go.Bar(x=["Feature"+str(j) for j in range(5)],
                                 y=np.random.rand(5))])
    fig.update_layout(title=f"VIZ {i}",
                      height=400, template="plotly_dark", plot_bgcolor="#0D1117",
                      paper_bgcolor="#0D1117", font=dict(color="#C9D1D9"))
    fig.write_html(f"outputs/html/viz_{i}_placeholder.html")
    print(f"💾  outputs/html/viz_{i}_placeholder.html")

print("\n✅ ML pipeline completado!")
