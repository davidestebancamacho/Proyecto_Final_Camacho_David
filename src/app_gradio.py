# ============================================================
#  MINI APP — PREDICTOR DE POPULARIDAD DE PELÍCULAS (TMDB)
#  Pregunta: ¿Qué características predicen mejor la popularidad?
#
#  Ejecución:
#    pip install gradio shap xgboost scikit-learn pandas numpy plotly
#    python app_gradio.py
# ============================================================

import pandas as pd
import numpy as np
import ast
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import os

warnings.filterwarnings("ignore")

# ── Constantes ───────────────────────────────────────────────
TOP_GENRES = ["Drama","Comedy","Thriller","Action",
              "Romance","Horror","Crime","Adventure"]
FEATURES   = (["log_budget","log_revenue","runtime",
                "vote_average","log_votes"] + TOP_GENRES)
TARGET     = "log_popularity"

NIVEL_LABELS = ["Muy baja","Baja","Media","Alta","Muy alta","Viral"]
NIVEL_EMOJIS = ["🥶","😐","🙂","🔥","🚀","🌟"]


# ════════════════════════════════════════════════════════════
#  ENTRENAMIENTO (se ejecuta al iniciar la app)
# ════════════════════════════════════════════════════════════

def entrenar_modelo(csv_path="movies_metadata.csv"):
    df = pd.read_csv(csv_path, low_memory=False)
    movies = df[["title","budget","revenue","runtime",
                 "vote_average","vote_count","popularity","genres"]].copy()

    for col in ["budget","revenue","popularity"]:
        movies[col] = pd.to_numeric(movies[col], errors="coerce")

    def parse_genres(g):
        try:
            return [x["name"] for x in ast.literal_eval(g)] if pd.notna(g) else []
        except Exception:
            return []

    movies["genre_list"] = movies["genres"].apply(parse_genres)
    for g in TOP_GENRES:
        movies[g] = movies["genre_list"].apply(lambda lst: int(g in lst))

    movies = movies.dropna(subset=["popularity"])
    for col in ["budget","revenue"]:
        movies[col] = movies[col].replace(0, np.nan)
        movies[col].fillna(movies[col].median(), inplace=True)
    movies["runtime"].fillna(movies["runtime"].median(), inplace=True)
    movies["vote_average"].fillna(movies["vote_average"].median(), inplace=True)
    movies["vote_count"].fillna(movies["vote_count"].median(), inplace=True)
    movies.drop_duplicates(subset=["title"], inplace=True)

    movies["log_budget"]     = np.log1p(movies["budget"])
    movies["log_revenue"]    = np.log1p(movies["revenue"])
    movies["log_votes"]      = np.log1p(movies["vote_count"])
    movies["log_popularity"] = np.log1p(movies["popularity"])

    X = movies[FEATURES]
    y = movies[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("m",  XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=42, verbosity=0))
    ])
    pipe.fit(X_train, y_train)

    pred   = pipe.predict(X_test)
    r2     = r2_score(y_test, pred)
    mae    = mean_absolute_error(y_test, pred)

    # SHAP sobre muestra del test
    n_shap = min(600, len(X_test))
    idx    = np.random.choice(len(X_test), n_shap, replace=False)
    X_sc   = pd.DataFrame(pipe.named_steps["sc"].transform(X_test),
                          columns=FEATURES)
    X_shap = X_sc.iloc[idx].reset_index(drop=True)
    fv_shap = X_test.iloc[idx].reset_index(drop=True)

    explainer   = shap.TreeExplainer(pipe.named_steps["m"])
    shap_values = explainer.shap_values(X_shap)
    shap_df     = pd.DataFrame(shap_values, columns=FEATURES)
    mean_shap   = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES)\
                    .sort_values(ascending=False)

    return pipe, r2, mae, mean_shap, shap_df, fv_shap, movies


print("⏳  Entrenando modelo XGBoost…")
PIPE, R2, MAE, MEAN_SHAP, SHAP_DF, FV_SHAP, MOVIES = entrenar_modelo()
print(f"✅  Modelo listo  |  R²={R2:.3f}  MAE={MAE:.3f}")


# ════════════════════════════════════════════════════════════
#  HELPERS DE VISUALIZACIÓN
# ════════════════════════════════════════════════════════════

def nivel_popularidad(pop_log):
    pop = np.expm1(pop_log)
    if pop < 5:     return 0
    elif pop < 15:  return 1
    elif pop < 40:  return 2
    elif pop < 100: return 3
    elif pop < 300: return 4
    else:           return 5

def gauge_chart(pop_log):
    pop   = float(np.expm1(pop_log))
    nivel = nivel_popularidad(pop_log)
    label = NIVEL_LABELS[nivel]
    emoji = NIVEL_EMOJIS[nivel]
    pct   = min(pop / 500, 1.0)

    steps = [
        dict(range=[0,5],   color="#1e3a5f"),
        dict(range=[5,15],  color="#1a5276"),
        dict(range=[15,40], color="#1f618d"),
        dict(range=[40,100],color="#2874a6"),
        dict(range=[100,300],color="#2e86c1"),
        dict(range=[300,500],color="#3498db"),
    ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(pop, 1),
        number=dict(suffix="  pts", font=dict(size=32, color="#ecf0f1")),
        title=dict(text=f"<b>{emoji} {label}</b><br><sup>Popularidad TMDB estimada</sup>",
                   font=dict(size=15, color="#bdc3c7")),
        gauge=dict(
            axis=dict(range=[0, 500], tickcolor="#7f8c8d",
                      tickfont=dict(color="#bdc3c7", size=10)),
            bar=dict(color="#3498db", thickness=0.25),
            bgcolor="#1a252f",
            bordercolor="#2c3e50",
            steps=steps,
            threshold=dict(
                line=dict(color="#e74c3c", width=3),
                thickness=0.8, value=pop
            )
        )
    ))
    fig.update_layout(
        height=260, margin=dict(t=60, b=20, l=30, r=30),
        paper_bgcolor="#1a252f", font=dict(color="#ecf0f1")
    )
    return fig

def shap_waterfall(input_vals):
    """Waterfall de contribuciones SHAP para la predicción actual."""
    x_in = pd.DataFrame([input_vals], columns=FEATURES)
    x_sc = pd.DataFrame(PIPE.named_steps["sc"].transform(x_in), columns=FEATURES)
    exp  = shap.TreeExplainer(PIPE.named_steps["m"])
    sv   = exp.shap_values(x_sc)[0]

    base  = float(exp.expected_value)
    pairs = sorted(zip(FEATURES, sv), key=lambda t: abs(t[1]), reverse=True)[:8]
    feats = [p[0] for p in pairs]
    vals  = [p[1] for p in pairs]

    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in vals]
    hover  = [f"{'↑' if v>0 else '↓'} {abs(v):.4f}" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=feats, orientation="h",
        marker_color=colors,
        text=hover, textposition="outside",
        textfont=dict(size=10, color="#bdc3c7"),
        hovertemplate="<b>%{y}</b><br>Contribución SHAP: %{x:.4f}<extra></extra>"
    ))
    fig.add_vline(x=0, line_color="#7f8c8d", line_width=1)
    fig.update_layout(
        title=dict(text="<b>¿Por qué esta predicción?</b> — Contribuciones SHAP",
                   font=dict(size=13, color="#ecf0f1")),
        height=320, margin=dict(t=50, b=20, l=20, r=60),
        paper_bgcolor="#1a252f", plot_bgcolor="#1e2d3d",
        xaxis=dict(gridcolor="#2c3e50", zerolinecolor="#7f8c8d",
                   tickfont=dict(color="#bdc3c7", size=10)),
        yaxis=dict(gridcolor="rgba(0,0,0,0)",
                   tickfont=dict(color="#ecf0f1", size=11))
    )
    return fig

def shap_global_bar():
    fig = go.Figure(go.Bar(
        x=MEAN_SHAP.values[::-1],
        y=MEAN_SHAP.index[::-1],
        orientation="h",
        marker=dict(
            color=list(range(len(MEAN_SHAP))),
            colorscale="Blues", showscale=False
        ),
        hovertemplate="<b>%{y}</b><br>|SHAP| medio: %{x:.4f}<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text="<b>Importancia global de variables (SHAP)</b>",
                   font=dict(size=13, color="#ecf0f1")),
        height=340, margin=dict(t=50, b=20, l=20, r=20),
        paper_bgcolor="#1a252f", plot_bgcolor="#1e2d3d",
        xaxis=dict(title="mean |SHAP|", gridcolor="#2c3e50",
                   tickfont=dict(color="#bdc3c7", size=10)),
        yaxis=dict(gridcolor="rgba(0,0,0,0)",
                   tickfont=dict(color="#ecf0f1", size=11))
    )
    return fig

def scatter_comparativo(pop_pred_log):
    sample = MOVIES.sample(min(1500, len(MOVIES)), random_state=1)
    pop_pred = float(np.expm1(pop_pred_log))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample["vote_count"].clip(upper=5000),
        y=np.expm1(sample["log_popularity"]).clip(upper=500),
        mode="markers",
        marker=dict(size=4, color="rgba(52,152,219,0.35)"),
        name="Dataset",
        hovertemplate="Votos: %{x:,.0f}<br>Popularidad: %{y:.1f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[pop_pred],
        mode="markers",
        marker=dict(size=16, color="#e74c3c", symbol="star",
                    line=dict(width=2, color="#fff")),
        name="Tu película",
    ))
    fig.add_hline(y=pop_pred, line_dash="dash",
                  line_color="#e74c3c", line_width=1.5,
                  annotation_text=f"  Tu película: {pop_pred:.1f}",
                  annotation_font_color="#e74c3c")

    fig.update_layout(
        title=dict(text="<b>Tu película vs el dataset</b>",
                   font=dict(size=13, color="#ecf0f1")),
        height=300, margin=dict(t=50, b=40, l=50, r=20),
        paper_bgcolor="#1a252f", plot_bgcolor="#1e2d3d",
        xaxis=dict(title="Votos (clip 5k)", gridcolor="#2c3e50",
                   tickfont=dict(color="#bdc3c7", size=10)),
        yaxis=dict(title="Popularidad TMDB", gridcolor="#2c3e50",
                   tickfont=dict(color="#bdc3c7", size=10)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#bdc3c7"))
    )
    return fig


# ════════════════════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL DE PREDICCIÓN
# ════════════════════════════════════════════════════════════

def predecir(budget, revenue, runtime, vote_average, vote_count,
             drama, comedy, thriller, action, romance, horror, crime, adventure):

    vals = {
        "log_budget":   np.log1p(float(budget)),
        "log_revenue":  np.log1p(float(revenue)),
        "runtime":      float(runtime),
        "vote_average": float(vote_average),
        "log_votes":    np.log1p(float(vote_count)),
        "Drama":    int(drama),
        "Comedy":   int(comedy),
        "Thriller": int(thriller),
        "Action":   int(action),
        "Romance":  int(romance),
        "Horror":   int(horror),
        "Crime":    int(crime),
        "Adventure":int(adventure),
    }

    x_in    = pd.DataFrame([[vals[f] for f in FEATURES]], columns=FEATURES)
    pop_log = float(PIPE.predict(x_in)[0])
    pop     = np.expm1(pop_log)
    nivel   = nivel_popularidad(pop_log)

    # Percentil aproximado en el dataset
    pct = int((MOVIES["log_popularity"] < pop_log).mean() * 100)

    # ── Insight textual ─────────────────────────────────────
    top_driver = MEAN_SHAP.index[0]
    x_sc = pd.DataFrame(PIPE.named_steps["sc"].transform(x_in), columns=FEATURES)
    sv   = shap.TreeExplainer(PIPE.named_steps["m"]).shap_values(x_sc)[0]
    shap_dict = dict(zip(FEATURES, sv))
    top_local = max(shap_dict, key=lambda k: abs(shap_dict[k]))

    insight = (
        f"**Popularidad estimada: {pop:.1f} pts** — {NIVEL_EMOJIS[nivel]} {NIVEL_LABELS[nivel]}\n\n"
        f"Tu película supera al **{pct}%** de las películas del dataset.\n\n"
        f"**Factor más influyente en esta predicción:** `{top_local}` "
        f"({'↑ empuja al alza' if shap_dict[top_local]>0 else '↓ reduce popularidad'})\n\n"
        f"**Consejo:** {'Incrementar la distribución (revenue) podría ser tu mayor palanca.' if top_local in ['log_revenue','log_budget'] else 'Generar más engagement temprano (votos/ratings) es la acción con mayor impacto según el modelo.'}"
    )

    input_vals = [vals[f] for f in FEATURES]

    return (
        gauge_chart(pop_log),
        shap_waterfall(input_vals),
        scatter_comparativo(pop_log),
        shap_global_bar(),
        insight
    )


# ════════════════════════════════════════════════════════════
#  INTERFAZ GRADIO
# ════════════════════════════════════════════════════════════

CSS = """
body, .gradio-container { background: #111827 !important; color: #e5e7eb !important; font-family: 'Inter', sans-serif; }
.gr-panel, .gr-box { background: #1f2937 !important; border: 1px solid #374151 !important; border-radius: 12px !important; }
h1 { color: #60a5fa !important; }
h3 { color: #93c5fd !important; }
label { color: #d1d5db !important; font-size: 13px !important; }
.gr-button-primary { background: #2563eb !important; border: none !important; border-radius: 8px !important; color: white !important; font-weight: 600 !important; }
.gr-button-primary:hover { background: #1d4ed8 !important; }
footer { display: none !important; }
.gr-markdown p { color: #d1d5db !important; line-height: 1.6; }
.gr-markdown strong { color: #60a5fa !important; }
.gr-markdown code { background: #374151; padding: 2px 6px; border-radius: 4px; color: #a5f3fc; }
"""

DESCRIPTION = """
## 🎬 Predictor de Popularidad de Películas — TMDB

**Pregunta del proyecto:** *¿Qué características de una película predicen mejor su popularidad en TMDB?*

Ingresa los datos de tu película y el modelo **XGBoost** (R²=0.74) te dará una estimación de popularidad
junto con una explicación SHAP de qué variables más influyeron en la predicción.
"""

with gr.Blocks(css=CSS, title="🎬 Movie Popularity Predictor") as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Row():
        # ── Panel izquierdo: inputs ──────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 💰 Datos económicos")
            budget = gr.Slider(
                0, 300_000_000, value=25_000_000, step=500_000,
                label="Presupuesto (USD)",
                info="Budget de producción en dólares"
            )
            revenue = gr.Slider(
                0, 2_000_000_000, value=80_000_000, step=1_000_000,
                label="Recaudación esperada (USD)",
                info="Revenue total estimado"
            )

            gr.Markdown("### 🎥 Características técnicas")
            runtime = gr.Slider(
                60, 240, value=105, step=1,
                label="Duración (minutos)"
            )
            vote_average = gr.Slider(
                1.0, 10.0, value=6.5, step=0.1,
                label="Calificación promedio esperada",
                info="Estimación de vote_average en TMDB"
            )
            vote_count = gr.Slider(
                0, 10_000, value=500, step=50,
                label="Número de votos esperados",
                info="Cuántas personas esperas que voten"
            )

            gr.Markdown("### 🎭 Géneros (selecciona todos los que apliquen)")
            with gr.Row():
                drama    = gr.Checkbox(label="Drama")
                comedy   = gr.Checkbox(label="Comedy")
                thriller = gr.Checkbox(label="Thriller")
                action   = gr.Checkbox(label="Action", value=True)
            with gr.Row():
                romance  = gr.Checkbox(label="Romance")
                horror   = gr.Checkbox(label="Horror")
                crime    = gr.Checkbox(label="Crime")
                adventure= gr.Checkbox(label="Adventure")

            btn = gr.Button("🔮 Predecir popularidad", variant="primary", size="lg")

        # ── Panel derecho: outputs ───────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Resultado de la predicción")
            insight_md = gr.Markdown("*Ajusta los parámetros y presiona Predecir…*")
            gauge_out   = gr.Plot(label="Medidor de popularidad")

            with gr.Row():
                waterfall_out = gr.Plot(label="¿Por qué esta predicción? (SHAP)")
                scatter_out   = gr.Plot(label="Tu película vs el dataset")

            global_shap_out = gr.Plot(label="Importancia global de variables")

    # ── Botón de predicción ──────────────────────────────────
    btn.click(
        fn=predecir,
        inputs=[budget, revenue, runtime, vote_average, vote_count,
                drama, comedy, thriller, action, romance, horror, crime, adventure],
        outputs=[gauge_out, waterfall_out, scatter_out, global_shap_out, insight_md]
    )

    # ── Predicción automática al cargar ─────────────────────
    demo.load(
        fn=predecir,
        inputs=[budget, revenue, runtime, vote_average, vote_count,
                drama, comedy, thriller, action, romance, horror, crime, adventure],
        outputs=[gauge_out, waterfall_out, scatter_out, global_shap_out, insight_md]
    )

    gr.Markdown("""
---
**Notas del modelo:**
- Entrenado con ~40K películas de TMDB (hasta ~2017).
- La popularidad TMDB es dinámica; estos valores son estimaciones basadas en patrones históricos.
- El modelo explica el **74.2%** de la varianza en popularidad (R² en test set).
""")


if __name__ == "__main__":
    demo.launch(
        share=False,          # True para generar link público temporal
        server_port=7860,
        show_error=True
    )
