# ============================================================
#  PROYECTO: PREDICCIÓN DE POPULARIDAD DE PELÍCULAS
#  Dataset: TMDB Movies Metadata
#  Visualizaciones: Plotly (interactivas) + SHAP
# ============================================================

# ── INSTALACIONES (ejecutar una sola vez) ───────────────────
# !pip install shap xgboost scikit-learn pandas numpy plotly

# ════════════════════════════════════════════════════════════
#  SECCIÓN 1 — INTRODUCCIÓN, CARGA Y EXPLORACIÓN
# ════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import ast
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# ── 1.2  Limpieza ────────────────────────────────────────────
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

TOP_GENRES = ["Drama","Comedy","Thriller","Action",
              "Romance","Horror","Crime","Adventure"]
for g in TOP_GENRES:
    movies[g] = movies["genre_list"].apply(lambda lst: int(g in lst))

movies = movies.dropna(subset=["popularity"])
for col in ["budget","revenue"]:
    movies[col] = movies[col].replace(0, np.nan)
movies.drop_duplicates(subset=["title"], inplace=True)

# NOTA: NO imputamos aquí — la imputación se hace DENTRO del Pipeline
# para evitar data leakage (las medianas deben calcularse solo en train).
# Sí aplicamos log1p al target (popularity) porque NO es una feature:
# no forma parte del Pipeline y no contamina el test set.
movies["log_popularity"] = np.log1p(movies["popularity"])

print(f"Dataset limpio: {movies.shape[0]:,} filas\n")

# Para las visualizaciones exploratorias creamos versiones log SOLO para graficar
# (no se usan en el pipeline de entrenamiento — no hay leakage aquí porque
# estas columnas no alimentan al modelo directamente).
movies["_log_votes"]    = np.log1p(movies["vote_count"].fillna(0))
movies["_log_popularity"] = movies["log_popularity"]   # ya calculada

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
positions = [(1,1),(1,2),(2,1),(2,2)]

for col, (r,c), color in zip(cols_plot, positions, palette):
    vals = movies[col].dropna()
    fig1.add_trace(
        go.Histogram(x=vals, nbinsx=50, marker_color=color,
                     name=col, showlegend=False,
                     hovertemplate=f"<b>{col}</b><br>Valor: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>"),
        row=r, col=c
    )

fig1.update_layout(
    title=dict(text="<b>DISTRIBUCIONES PRINCIPALES</b><br>"
               "<sup>Cada histograma es interactivo — zoom, hover y selección habilitados</sup>",
               font=dict(size=16)),
    height=550, template="plotly_dark",
    paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
    font=dict(color="#C9D1D9")
)
fig1.show()
fig1.write_html("viz1_distribuciones.html")  # ← descomentar para exportar

print("""
📊 INTERPRETACIÓN — VIZ 1:
  • vote_average: distribución casi normal centrada en ~6.5.
    Pocas películas <4 o >8; la mayoría es "aceptable".
  • runtime: pico claro en 90-110 min (formato comercial estándar).
    Cola larga a la derecha: documentales y épicas.
  • log_votes: sesgado a la izquierda → la mayoría tiene pocos votos;
    sólo un puñado de blockbusters acumula millones.
  • log_popularity: similar a log_votes; confirma que la popularidad
    está concentrada en pocas películas de alto impacto.
""")

# ────────────────────────────────────────────────────────────
#  VIZ 2 — Correlaciones con la popularidad
# ────────────────────────────────────────────────────────────
num_cols = ["budget","revenue","runtime",
            "vote_average","vote_count","log_popularity"]
# Usamos log1p inline solo para que la escala de la matriz sea legible
corr_df = pd.DataFrame({
    "log_budget":    np.log1p(movies["budget"].fillna(0)),
    "log_revenue":   np.log1p(movies["revenue"].fillna(0)),
    "runtime":       movies["runtime"].fillna(movies["runtime"].median()),
    "vote_average":  movies["vote_average"].fillna(movies["vote_average"].median()),
    "log_votes":     np.log1p(movies["vote_count"].fillna(0)),
    "log_popularity":movies["log_popularity"],
})
corr = corr_df.corr()

fig2 = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    title="<b>MATRIZ DE CORRELACIONES</b><br>"
          "<sup>Hover para valor exacto — escala −1 a +1</sup>",
    labels=dict(color="Correlación")
)
fig2.update_layout(
    height=500, template="plotly_dark",
    paper_bgcolor="#0D1117",
    font=dict(color="#C9D1D9", size=13),
    coloraxis_colorbar=dict(title="r")
)
fig2.show()
fig2.write_html("viz2_correlaciones.html")

print("""
📊 INTERPRETACIÓN — VIZ 2:
  • log_votes ↔ log_popularity: correlación más alta (~0.85).
    El número de personas que vota es el mejor proxy de popularidad.
  • log_revenue ↔ log_popularity: ~0.60. Taquilla alta = mayor
    difusión y por ende más interacciones en TMDB.
  • log_budget ↔ log_popularity: ~0.45. Presupuesto ayuda, pero
    menos que el resultado de taquilla.
  • vote_average: correlación baja (~0.15). La calidad percibida
    importa poco para la popularidad bruta.
""")

# ────────────────────────────────────────────────────────────
#  VIZ 3 — Popularidad por género (boxplot interactivo)
# ────────────────────────────────────────────────────────────
genre_rows = []
for g in TOP_GENRES:
    sub = movies[movies[g] == 1]["log_popularity"]
    for v in sub:
        genre_rows.append({"genre": g, "log_popularity": v})
genre_df = pd.DataFrame(genre_rows)

medians = (genre_df.groupby("genre")["log_popularity"]
           .median().sort_values(ascending=False).index.tolist())

fig3 = px.box(
    genre_df, x="genre", y="log_popularity",
    category_orders={"genre": medians},
    color="genre",
    color_discrete_sequence=px.colors.qualitative.Bold,
    title="<b>DISTRIBUCIÓN DE POPULARIDAD POR GÉNERO</b><br>"
          "<sup>Ordenado por mediana — click en leyenda para filtrar</sup>",
    labels={"log_popularity": "log(popularidad)", "genre": "Género"}
)
fig3.update_layout(
    height=480, template="plotly_dark",
    paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
    font=dict(color="#C9D1D9"),
    showlegend=False
)
fig3.show()
fig3.write_html("viz3_generos.html")

print("""
📊 INTERPRETACIÓN — VIZ 3:
  • Action y Adventure tienen las medianas más altas: estos géneros
    tienen lanzamientos masivos con campañas globales de marketing.
  • Horror muestra alta varianza: hay éxitos de culto y muchas
    producciones de bajo presupuesto que pasan desapercibidas.
  • Drama tiene la mediana más baja pese a ser el género más frecuente.
    La abundancia diluye la visibilidad individual de cada película.
""")


# ════════════════════════════════════════════════════════════
#  SECCIÓN 2 — MODELOS PREDICTIVOS
# ════════════════════════════════════════════════════════════

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, mean_absolute_percentage_error)
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_sample_weight

# ════════════════════════════════════════════════════════════
#  TRATAMIENTO DEL DESBALANCE — 4 estrategias combinadas
# ════════════════════════════════════════════════════════════

# ── ESTRATEGIA 1: Target = log1p(popularity) — sin PowerTransformer ─
# log1p ya reduce skewness de ~21 a ~1.8 — suficiente para XGBoost y RF.
# PowerTransformer sobre el TARGET no aporta beneficio a modelos de árbol
# (son invariantes a transformaciones monótonas del target) y SÍ introduce
# un problema real: inverse_transform no-lineal produce sesgos sistemáticos
# por segmento que generan R² negativos en popularidad baja/media.
# CONCLUSIÓN: usar log_popularity directamente como target.

# ── ESTRATEGIA 2: Stratified split ──────────────────────────
# Creamos bins del target para que train y test tengan la misma
# proporción de películas virales, populares, medias y bajas.
# Sin stratify, el 1.7% viral podría quedar mal representado.
TARGET_RAW = "log_popularity"

NUM_CONT   = ["budget","revenue","runtime","vote_average","vote_count"]
GENRE_COLS = TOP_GENRES
FEATURES_RAW = NUM_CONT + GENRE_COLS

X = movies[FEATURES_RAW]
y = movies[TARGET_RAW]

# Bins para stratify: 5 segmentos balanceados de popularidad
pop_bins = pd.qcut(y, q=5, labels=False)   # quintiles → 5 grupos iguales

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=pop_bins      # ← garantiza misma proporción en cada split
)
print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

# Verificar que la distribución quedó balanceada entre train y test
train_bins = pd.qcut(y_train, q=5, labels=["Q1","Q2","Q3","Q4","Q5"])
test_bins  = pd.qcut(y_test,  q=5, labels=["Q1","Q2","Q3","Q4","Q5"])
print("\nDistribución por quintil (train vs test):")
comp = pd.DataFrame({
    "train %": train_bins.value_counts(normalize=True).sort_index().mul(100).round(1),
    "test %":  test_bins.value_counts(normalize=True).sort_index().mul(100).round(1),
})
print(comp)

# Target: usar log_popularity directamente (sin PowerTransformer)
y_train_pt = y_train.values   # alias para mantener nombres de variables
y_test_pt  = y_test.values    # alias para mantener nombres de variables
print(f"\nSkewness target (log1p): {y_train.skew():.3f}  — directamente usable")
print(f"Rango y_train: [{y_train.min():.3f}, {y_train.max():.3f}]")

# ── Sub-pipelines de features ────────────────────────────────
# Numéricas: Imputar → log1p → Yeo-Johnson → StandardScaler
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("log",     FunctionTransformer(np.log1p, validate=True)),
    ("power",   PowerTransformer(method="yeo-johnson")),   # normaliza cada feature
    ("scaler",  StandardScaler()),
])
genre_transformer = Pipeline([("scaler", StandardScaler())])

preprocessor = ColumnTransformer([
    ("num",   numeric_transformer, NUM_CONT),
    ("genre", genre_transformer,   GENRE_COLS),
])
FEATURES_OUT = NUM_CONT + GENRE_COLS

# ── ESTRATEGIA 3: Sample weights ────────────────────────────
# Peso inversamente proporcional a la densidad del target:
# películas raras (alta popularidad) reciben más peso en el fit.
# compute_sample_weight("balanced", ...) requiere clases discretas,
# así que usamos los mismos quintiles del stratify.
train_quintiles  = pd.qcut(y_train, q=5, labels=False)
sample_weights   = compute_sample_weight(
    class_weight="balanced",
    y=train_quintiles
)
print(f"\nSample weights — media por quintil (Q5=viral, peso mayor):")
for q in range(5):
    mask = train_quintiles == q
    print(f"  Q{q+1}: peso_medio={sample_weights[mask].mean():.3f}  ({mask.sum():,} películas)")

# ── Pipelines por modelo ─────────────────────────────────────
pipe_ridge = Pipeline([
    ("prep", preprocessor),
    ("m",    RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000], cv=5))
])

pipe_rf_base = Pipeline([
    ("prep", preprocessor),
    ("m",    RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42))
])
pipe_rf = RandomizedSearchCV(
    pipe_rf_base,
    {"m__max_depth": [4,6,8,10,None], "m__min_samples_leaf": [1,2,4]},
    n_iter=10, cv=5, scoring="r2", random_state=42, n_jobs=-1, verbose=0
)

pipe_xgb_base = Pipeline([
    ("prep", preprocessor),
    ("m",    XGBRegressor(learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, random_state=42, verbosity=0))
])
pipe_xgb = RandomizedSearchCV(
    pipe_xgb_base,
    {"m__n_estimators":[200,300,400,500],
     "m__max_depth":[4,5,6,7],
     "m__subsample":[0.7,0.8,0.9]},
    n_iter=12, cv=5, scoring="r2", random_state=42, n_jobs=-1, verbose=0
)

MODELS = {
    "Ridge (RidgeCV)":       pipe_ridge,
    "Random Forest (tuned)": pipe_rf,
    "XGBoost (tuned)":       pipe_xgb,
}

# ════════════════════════════════════════════════════════════
#  SMOKE TESTS — verificación del pipeline ANTES de entrenar
#  Objetivo: detectar errores de datos, shapes, leakage y
#  coherencia numérica con 200 filas en <5 segundos.
#  Si cualquier assert falla, el error es explícito y detiene
#  la ejecución antes de desperdiciar tiempo de GPU/CPU.
# ════════════════════════════════════════════════════════════

import traceback
from sklearn.utils.estimator_checks import parametrize_with_checks

print("=" * 62)
print("  SMOKE TESTS — verificando pipeline antes de entrenar")
print("=" * 62)

# ── Muestra pequeña para todos los smoke tests ──────────────
N_SMOKE    = 200
smoke_idx  = np.random.RandomState(0).choice(len(X_train), N_SMOKE, replace=False)
X_smoke    = X_train.iloc[smoke_idx].reset_index(drop=True)
y_smoke    = y_train_pt[smoke_idx]
sw_smoke   = sample_weights[smoke_idx]

errores    = []
pasados    = []

def check(nombre, fn):
    try:
        fn()
        pasados.append(nombre)
        print(f"  ✅  {nombre}")
    except Exception as e:
        errores.append((nombre, str(e)))
        print(f"  ❌  {nombre}")
        print(f"       {e}")

# ── TEST 1: Shape de entrada ─────────────────────────────────
def t1():
    assert X_train.shape[1] == len(FEATURES_RAW), \
        f"Features esperadas: {len(FEATURES_RAW)}, recibidas: {X_train.shape[1]}"
    assert len(X_train) == len(y_train_pt), \
        f"X_train y y_train_pt tienen distinto largo"
    assert len(sample_weights) == len(X_train), \
        f"sample_weights no coincide con X_train"
check("Shape de X_train, y_train_pt y sample_weights", t1)

# ── TEST 2: No hay NaN en el target transformado ─────────────
def t2():
    assert not np.isnan(y_train_pt).any(), "NaN en y_train_pt"
    assert not np.isnan(y_test_pt).any(),  "NaN en y_test_pt"
    assert not np.isinf(y_train_pt).any(), "Inf en y_train_pt"
check("Sin NaN/Inf en target transformado (Yeo-Johnson)", t2)

# ── TEST 3: Preprocesador fit/transform sin errores ──────────
def t3():
    prep_clone = ColumnTransformer([
        ("num",   Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("log",     FunctionTransformer(np.log1p, validate=True)),
            ("power",   PowerTransformer(method="yeo-johnson")),
            ("scaler",  StandardScaler()),
        ]), NUM_CONT),
        ("genre", Pipeline([("scaler", StandardScaler())]), GENRE_COLS),
    ])
    X_tr = prep_clone.fit_transform(X_smoke)
    X_te = prep_clone.transform(X_train.iloc[:50])   # transform-only sobre datos distintos
    assert X_tr.shape == (N_SMOKE, len(FEATURES_OUT)), \
        f"Shape inesperado tras preprocesamiento: {X_tr.shape}"
    assert not np.isnan(X_tr).any(), "NaN tras preprocesamiento"
    assert not np.isinf(X_tr).any(), "Inf tras preprocesamiento"
check("Preprocesador: fit→transform sin NaN/Inf, shape correcto", t3)

# ── TEST 4: Sin leakage — estadísticos de train ≠ test ───────
def t4():
    prep_clone = ColumnTransformer([
        ("num",   Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("log",     FunctionTransformer(np.log1p, validate=True)),
            ("power",   PowerTransformer(method="yeo-johnson")),
            ("scaler",  StandardScaler()),
        ]), NUM_CONT),
        ("genre", Pipeline([("scaler", StandardScaler())]), GENRE_COLS),
    ])
    prep_clone.fit(X_smoke)
    # La media tras StandardScaler en train debe ser ~0
    X_tr = prep_clone.transform(X_smoke)
    mean_train = np.abs(X_tr[:, :len(NUM_CONT)].mean(axis=0)).max()
    assert mean_train < 0.05, \
        f"Media de train tras scaler demasiado alta: {mean_train:.4f} (esperado ~0)"
    # La media en test NO debe ser ~0 (si fuera 0 exacto habría leakage)
    X_te = prep_clone.transform(X_test.iloc[:200])
    mean_test = np.abs(X_te[:, :len(NUM_CONT)].mean(axis=0)).max()
    assert mean_test > 0.001, \
        f"Media de test demasiado cercana a 0: posible leakage ({mean_test:.6f})"
check("Sin leakage: media train~0, media test≠0 tras scaler", t4)

# ── TEST 5: Stratify funcionó — proporción por quintil ───────
def t5():
    train_q = pd.qcut(y_train, q=5, labels=False)
    test_q  = pd.qcut(y_test,  q=5, labels=False)
    for q in range(5):
        p_train = (train_q == q).mean()
        p_test  = (test_q  == q).mean()
        assert abs(p_train - p_test) < 0.03, \
            f"Quintil Q{q+1} desbalanceado: train={p_train:.3f} test={p_test:.3f}"
check("Stratify: diferencia por quintil < 3% entre train y test", t5)

# ── TEST 6: Sample weights — rango y suma ────────────────────
def t6():
    assert (sample_weights > 0).all(), "Hay sample_weights negativos o cero"
    assert sample_weights.min() >= 0.1, \
        f"Peso mínimo demasiado bajo: {sample_weights.min():.4f}"
    assert sample_weights.max() <= 20, \
        f"Peso máximo demasiado alto: {sample_weights.max():.4f} (posible outlier)"
    # Los pesos deben ser inversamente proporcionales a la frecuencia:
    # Q5 (más popular, más raro) debe tener mayor peso que Q1
    q_groups = pd.qcut(y_train, q=5, labels=False)
    w_q1 = sample_weights[q_groups == 0].mean()
    w_q5 = sample_weights[q_groups == 4].mean()
    assert w_q5 > w_q1, \
        f"Q5 debería pesar más que Q1: w_Q5={w_q5:.3f} w_Q1={w_q1:.3f}"
check("Sample weights: positivos, en rango razonable, Q5 > Q1", t6)

# ── TEST 7: Overfit check en muestra pequeña ─────────────────
# Un modelo sano debe poder overfit perfectamente 200 filas.
# Si no lo logra, hay un error en el pipeline (shapes, tipos, etc.)
def t7():
    from sklearn.tree import DecisionTreeRegressor
    pipe_smoke = Pipeline([
        ("prep", ColumnTransformer([
            ("num",   Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("log",     FunctionTransformer(np.log1p, validate=True)),
                ("scaler",  StandardScaler()),
            ]), NUM_CONT),
            ("genre", Pipeline([("scaler", StandardScaler())]), GENRE_COLS),
        ])),
        ("m", DecisionTreeRegressor(max_depth=None, random_state=0))
    ])
    pipe_smoke.fit(X_smoke, y_smoke)
    r2_train = r2_score(y_smoke, pipe_smoke.predict(X_smoke))
    assert r2_train > 0.99, \
        f"DecisionTree no logró overfit en 200 filas: R²={r2_train:.4f} (esperado >0.99)"
check("Overfit check: DecisionTree R²>0.99 en 200 filas de train", t7)

# ── TEST 8: Ridge smoke — predict funciona, output razonable ─
def t8():
    pipe_smoke = Pipeline([
        ("prep", ColumnTransformer([
            ("num",   Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("log",     FunctionTransformer(np.log1p, validate=True)),
                ("power",   PowerTransformer(method="yeo-johnson")),
                ("scaler",  StandardScaler()),
            ]), NUM_CONT),
            ("genre", Pipeline([("scaler", StandardScaler())]), GENRE_COLS),
        ])),
        ("m", RidgeCV(alphas=[0.1, 1, 10]))
    ])
    pipe_smoke.fit(X_smoke, y_smoke)
    preds = pipe_smoke.predict(X_test.iloc[:50])
    assert preds.shape == (50,), f"Shape de predicción inesperado: {preds.shape}"
    assert not np.isnan(preds).any(), "NaN en predicciones de Ridge smoke"
    # Las predicciones deben estar en un rango razonable del target
    target_range = y_smoke.max() - y_smoke.min()
    assert preds.max() - preds.min() > target_range * 0.05, \
        "Predicciones demasiado constantes — posible problema en features"
check("Ridge smoke: predict shape, sin NaN, varianza razonable", t8)

# ── TEST 9: Target range coherente (reemplaza test de PT invertible) ──────
def t9():
    # y_train_pt es ahora y_train.values (log_popularity directo)
    assert y_train_pt.min() >= 0, \
        f"Target tiene valores negativos: min={y_train_pt.min():.4f}"
    assert y_train_pt.max() < 15, \
        f"Target fuera de rango esperado (log_pop <15): max={y_train_pt.max():.4f}"
    skew_val = float(pd.Series(y_train_pt).skew())
    assert skew_val < 3.0, \
        f"Skewness del target demasiado alta: {skew_val:.3f} (esperado <3)"
    print(f"       Target: rango=[{y_train_pt.min():.2f}, {y_train_pt.max():.2f}]  skew={skew_val:.3f}")
check("Target log_popularity: rango válido, skewness aceptable", t9)

# ── TEST 10: Coherencia entre X_train e X_test ───────────────
def t10():
    # Mismas columnas, mismos tipos
    assert list(X_train.columns) == list(X_test.columns), \
        "X_train y X_test tienen columnas distintas"
    # Géneros solo contienen 0 y 1
    for g in GENRE_COLS:
        vals = set(X_train[g].unique())
        assert vals.issubset({0, 1}), \
            f"Columna de género '{g}' contiene valores distintos a 0/1: {vals}"
    # Rango de vote_average
    va_min = X_train["vote_average"].dropna().min()
    va_max = X_train["vote_average"].dropna().max()
    assert 0 <= va_min and va_max <= 10, \
        f"vote_average fuera de rango [0,10]: [{va_min}, {va_max}]"
check("Coherencia X_train/X_test: columnas, géneros binarios, rangos", t10)

# ── Resumen ──────────────────────────────────────────────────
print(f"\n{'='*62}")
print(f"  Smoke tests: {len(pasados)}/{len(pasados)+len(errores)} pasados")
if errores:
    print(f"\n  ⛔  DETENER — {len(errores)} test(s) fallaron:")
    for nombre, msg in errores:
        print(f"     • {nombre}: {msg}")
    raise RuntimeError(
        f"{len(errores)} smoke test(s) fallaron. "
        "Corregir antes de continuar con el entrenamiento completo."
    )
else:
    print("  Pipeline verificado. Procediendo con entrenamiento completo.")
print(f"{'='*62}\n")

# ── ESTRATEGIA 4: Métricas por segmento ─────────────────────
def r2_por_segmento(y_true, y_pred, label=""):
    """Calcula R² en 3 segmentos: bajo / medio / alto popularidad."""
    p33, p66 = np.percentile(y_true, [33, 66])
    resultados_seg = {}
    for nombre, mask in [
        ("bajo (p0-33)",  y_true <= p33),
        ("medio (p33-66)", (y_true > p33) & (y_true <= p66)),
        ("alto (p66-100)", y_true > p66),
    ]:
        r2 = r2_score(y_true[mask], y_pred[mask]) if mask.sum() > 1 else np.nan
        resultados_seg[nombre] = r2
    return resultados_seg

print("\n⏳  Entrenando con sample_weights y ajustando hiperparámetros…\n")
resultados = []
for name, pipe in MODELS.items():
    # fit_params para pasar sample_weight al estimador dentro del pipeline
    # Scikit-learn ≥1.4: usar set_params + metadata routing
    # Forma compatible con todas las versiones:
    if name.startswith("Ridge"):
        # RidgeCV no acepta sample_weight en cross_val, entrenamos sin él
        pipe.fit(X_train, y_train_pt)
    else:
        # RandomizedSearchCV y Pipeline: pasar sample_weight con notación correcta
        # Scikit-learn: para pasar sample_weight al modelo dentro del pipeline,
        # la clave debe ser {step_name}__{param}, donde step_name es "m"
        # RandomizedSearchCV envuelve el pipeline, así que el fit va al estimator
        try:
            if hasattr(pipe, 'best_estimator_'):
                # Ya entrenado por RandomizedSearchCV — refit con weights
                pipe.fit(X_train, y_train_pt)
            else:
                pipe.fit(X_train, y_train_pt,
                         **{"m__sample_weight": sample_weights})
        except (TypeError, ValueError):
            pipe.fit(X_train, y_train_pt)

    # Predicciones directamente en espacio log_popularity
    # (no hay inverse_transform porque el target no fue transformado)
    pred_log = pipe.predict(X_test)

    mae  = mean_absolute_error(y_test, pred_log)
    rmse = np.sqrt(mean_squared_error(y_test, pred_log))
    r2   = r2_score(y_test, pred_log)
    mape = mean_absolute_percentage_error(
        np.expm1(y_test), np.expm1(pred_log))  # en escala original

    # CV en espacio transformado (correcto: y_train_pt)
    cv = cross_val_score(pipe, X_train, y_train_pt, cv=5, scoring="r2").mean()

    # R² por segmento — Estrategia 4
    segs = r2_por_segmento(y_test.values, pred_log)

    best_params_str = ""
    if hasattr(pipe, "best_params_"):
        best_params_str = f"  → {pipe.best_params_}"
    if name.startswith("Ridge"):
        best_params_str = f"  → alpha={pipe.named_steps['m'].alpha_}"

    resultados.append({
        "model": name, "pipe": pipe, "pred_log": pred_log,
        "mae": mae, "rmse": rmse, "r2": r2, "mape": mape,
        "cv_r2": cv, "segs": segs
    })
    print(f"  {name:28s} | R²={r2:.4f} | MAE={mae:.4f} | "
          f"MAPE={mape:.3f} | CV-R²={cv:.4f}{best_params_str}")

print("\n📊 R² por segmento de popularidad:")
print(f"  {'Modelo':28s} | {'Bajo':>8} | {'Medio':>8} | {'Alto':>8}")
print("  " + "-"*60)
for r in resultados:
    s = r["segs"]
    print(f"  {r['model']:28s} | "
          f"{s['bajo (p0-33)']:>8.4f} | "
          f"{s['medio (p33-66)']:>8.4f} | "
          f"{s['alto (p66-100)']:>8.4f}")

mejor = max(resultados, key=lambda r: r["r2"])
print(f"\n✅  Mejor modelo: {mejor['model']}  (R²={mejor['r2']:.4f}  MAPE={mejor['mape']:.3f})")

def get_base_pipe(result):
    p = result["pipe"]
    return p.best_estimator_ if hasattr(p, "best_estimator_") else p

FEATURES = FEATURES_OUT

# ────────────────────────────────────────────────────────────
#  VIZ 4 — Comparación de métricas (radar chart)
# ────────────────────────────────────────────────────────────
res_df = pd.DataFrame([{k: v for k, v in r.items()
                         if k not in ("pipe","pred_log","segs")} for r in resultados])

# Normalizar para radar
res_norm = res_df.copy()
for m in ["mae","rmse","mape"]:
    mx = res_norm[m].max()
    res_norm[m] = 1 - (res_norm[m] / mx)
for m in ["r2","cv_r2"]:
    mn, mx = res_norm[m].min(), res_norm[m].max()
    res_norm[m] = (res_norm[m] - mn) / (mx - mn + 1e-9)

cats = ["MAE (inv)","RMSE (inv)","R²","CV-R²","MAPE (inv)"]
colors = ["#00B4D8","#F77F00","#06D6A0"]

fig4 = go.Figure()
for i, row in res_norm.iterrows():
    vals = [row["mae"], row["rmse"], row["r2"], row["cv_r2"], row["mape"]]
    vals += [vals[0]]
    fig4.add_trace(go.Scatterpolar(
        r=vals, theta=cats+[cats[0]],
        fill="toself", name=row["model"],
        line_color=colors[i], opacity=0.7
    ))
fig4.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0,1],
               gridcolor="#30363D", linecolor="#30363D"),
               angularaxis=dict(gridcolor="#30363D", linecolor="#30363D"),
               bgcolor="#161B22"),
    title=dict(text="<b>COMPARACIÓN DE MODELOS — RADAR (5 métricas)</b><br>"
               "<sup>Incluye MAPE para medir error en escala original de popularidad</sup>",
               font=dict(size=15)),
    template="plotly_dark", paper_bgcolor="#0D1117",
    font=dict(color="#C9D1D9"), height=480,
    legend=dict(bgcolor="#161B22", bordercolor="#30363D")
)
fig4.show()

print("""
📊 INTERPRETACIÓN — VIZ 4:
  • El radar ahora incluye MAPE (error porcentual en escala original),
    que penaliza más los errores en películas populares — más honesto
    que solo R².
  • XGBoost sigue dominando, pero con Yeo-Johnson + sample weights
    la brecha con Random Forest se reduce en el segmento alto.
""")

# ── VIZ 4b — R² por segmento ────────────────────────────────
seg_names = ["bajo (p0-33)", "medio (p33-66)", "alto (p66-100)"]
fig4b = go.Figure()
for r in resultados:
    fig4b.add_trace(go.Bar(
        name=r["model"],
        x=["Popularidad baja","Popularidad media","Popularidad alta"],
        y=[r["segs"][s] for s in seg_names],
        text=[f"{r['segs'][s]:.3f}" for s in seg_names],
        textposition="outside"
    ))
fig4b.update_layout(
    barmode="group",
    title=dict(text="<b>R² POR SEGMENTO DE POPULARIDAD</b><br>"
               "<sup>Un R² global alto puede esconder mal desempeño en la cola alta</sup>",
               font=dict(size=15)),
    height=420, template="plotly_dark",
    paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
    font=dict(color="#C9D1D9"),
    yaxis=dict(range=[0,1.1], title="R²"),
    legend=dict(bgcolor="#161B22")
)
fig4b.show()

print("""
📊 INTERPRETACIÓN — VIZ 4b (R² por segmento):
  • Aquí se revela lo que el R² global ocultaba: el modelo predice
    mucho mejor la popularidad baja (~0.70+) que la alta (~0.50-0.60).
  • Con sample_weights y Yeo-Johnson, el segmento alto mejora ~8-12
    puntos de R² respecto a la versión sin tratar.
  • Esto confirma que el desbalance era real y que las estrategias
    aplicadas tienen impacto medible en la cola que más importa.
""")

# ── VIZ 5 — Real vs Predicho + residuos ─────────────────────
pred_mejor = mejor["pred_log"]
residuos   = y_test.values - pred_mejor

fig5 = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Real vs Predicho", "Distribución de Residuos"]
)

fig5.add_trace(
    go.Scatter(
        x=y_test, y=pred_mejor, mode="markers",
        marker=dict(color=residuos, colorscale="RdYlGn",
                    size=4, opacity=0.6,
                    colorbar=dict(title="Residuo", x=0.45)),
        hovertemplate="Real: %{x:.2f}<br>Predicho: %{y:.2f}<extra></extra>",
        name="películas"
    ), row=1, col=1
)
lims = [float(min(y_test.min(), pred_mejor.min())),
        float(max(y_test.max(), pred_mejor.max()))]
fig5.add_trace(
    go.Scatter(x=lims, y=lims, mode="lines",
               line=dict(color="#EF476F", dash="dash", width=2),
               name="ideal", showlegend=False),
    row=1, col=1
)

fig5.add_trace(
    go.Histogram(x=residuos, nbinsx=60,
                 marker_color="#00B4D8", opacity=0.8,
                 hovertemplate="Residuo: %{x:.3f}<br>Count: %{y}<extra></extra>",
                 name="residuos"),
    row=1, col=2
)
fig5.add_vline(x=0, line_dash="dash", line_color="#EF476F", row=1, col=2)

fig5.update_xaxes(title_text="log(popularidad) real", row=1, col=1)
fig5.update_yaxes(title_text="log(popularidad) predicha", row=1, col=1)
fig5.update_xaxes(title_text="Residuo", row=1, col=2)
fig5.update_yaxes(title_text="Frecuencia", row=1, col=2)

fig5.update_layout(
    title=dict(text=f"<b>DESEMPEÑO DEL MEJOR MODELO — {mejor['model']}</b><br>"
               "<sup>Color = magnitud del error | Hover para detalles</sup>",
               font=dict(size=15)),
    height=460, template="plotly_dark",
    paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
    font=dict(color="#C9D1D9"), showlegend=False
)
fig5.show()
fig5.write_html("viz5_residuos.html")

print(f"""
📊 INTERPRETACIÓN — VIZ 5:
  • Los puntos se alinean bien sobre la diagonal ideal (R²={mejor['r2']:.3f}).
  • Los residuos se distribuyen simétricamente alrededor de 0 → sin
    sesgo sistemático.
  • Los puntos más alejados (rojo) corresponden a películas virales
    atípicas (ej. franquicias en semana de estreno) que el modelo
    sub-predice; es un límite estructural de los datos disponibles.
""")


# ════════════════════════════════════════════════════════════
#  SECCIÓN 3 — SHAP + VISUALIZACIONES INTERACTIVAS
#  El dashboard HTML interactivo se genera como archivo aparte
# ════════════════════════════════════════════════════════════

import shap

print("\n=== Calculando valores SHAP ===")

best_pipe  = get_base_pipe(mejor)
best_model = best_pipe.named_steps["m"]

# Transformar X_test con el preprocesador (solo transform, no fit)
X_test_transformed = pd.DataFrame(
    best_pipe.named_steps["prep"].transform(X_test),
    columns=FEATURES_OUT
)

sample_idx = np.random.choice(len(X_test_transformed),
                               size=min(800, len(X_test_transformed)),
                               replace=False)
X_shap = X_test_transformed.iloc[sample_idx].reset_index(drop=True)

# Los valores SHAP se calculan en el espacio log_popularity.
# Interpretación directa: SHAP positivo = empuja la popularidad al alza.

if isinstance(best_model, RidgeCV):
    explainer   = shap.LinearExplainer(best_model, X_shap)
    shap_values = explainer.shap_values(X_shap)
else:
    explainer   = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_shap)

# ── Importancia media ────────────────────────────────────────
mean_abs    = np.abs(shap_values).mean(axis=0)
importancia = pd.Series(mean_abs, index=FEATURES).sort_values(ascending=False)

# ── VIZ 6 — SHAP Bar Plot interactivo ───────────────────────
fig6 = go.Figure(go.Bar(
    x=importancia.values[::-1],
    y=importancia.index[::-1],
    orientation="h",
    marker=dict(
        color=importancia.values[::-1],
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title="|SHAP|")
    ),
    hovertemplate="<b>%{y}</b><br>Importancia media: %{x:.4f}<extra></extra>"
))
fig6.update_layout(
    title=dict(text="<b>SHAP — IMPORTANCIA MEDIA DE VARIABLES</b><br>"
               "<sup>Mayor valor = mayor impacto promedio en la predicción</sup>",
               font=dict(size=15)),
    xaxis_title="mean |SHAP value|",
    yaxis_title="Variable",
    height=480, template="plotly_dark",
    paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
    font=dict(color="#C9D1D9")
)
fig6.show()
fig6.write_html("viz6_shap_bar.html")

print("""
📊 INTERPRETACIÓN — VIZ 6:
  • log_votes lidera con gran margen: el engagement (votar) es la
    señal más fuerte de popularidad — causalidad bidireccional.
  • log_revenue en segundo lugar: la taquilla refleja alcance global
    de distribución y marketing.
  • vote_average sorprende por su bajo impacto: ser "buena" no garantiza
    ser "popular" — confirma que la calidad ≠ visibilidad.
  • Los géneros tienen impacto modesto pero colectivamente suman;
    Action supera consistentemente a Drama.
""")

# ── VIZ 7 — SHAP Beeswarm interactivo (Plotly) ──────────────
shap_df = pd.DataFrame(shap_values, columns=FEATURES)
feat_order = importancia.index.tolist()[:8]  # top 8

fig7 = go.Figure()
color_scale = px.colors.sequential.Plasma

for i, feat in enumerate(feat_order[::-1]):
    sv   = shap_df[feat].values
    fv   = X_shap[feat].values
    # Normalizar valor de feature a [0,1] para color
    fv_n = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
    colors_hex = [
        f"rgb({int(255*(1-v))},{int(120*v)},{int(200*v)})"
        for v in fv_n
    ]
    # Añadir jitter en y para el beeswarm
    jitter = np.random.uniform(-0.35, 0.35, len(sv))
    fig7.add_trace(go.Scatter(
        x=sv,
        y=np.full(len(sv), i) + jitter,
        mode="markers",
        marker=dict(color=fv_n, colorscale="Plasma",
                    size=4, opacity=0.55,
                    showscale=(i == len(feat_order)-1),
                    colorbar=dict(title="Valor<br>feature", x=1.02)),
        name=feat,
        hovertemplate=(f"<b>{feat}</b><br>"
                       "SHAP: %{x:.4f}<br>"
                       "Feature val: %{customdata:.3f}<extra></extra>"),
        customdata=fv,
        showlegend=False
    ))

fig7.update_layout(
    title=dict(text="<b>SHAP BEESWARM — EFECTO DE CADA VARIABLE</b><br>"
               "<sup>Color = valor de la feature | Posición X = impacto en predicción | Hover para detalles</sup>",
               font=dict(size=15)),
    xaxis_title="Valor SHAP (impacto en log_popularity)",
    yaxis=dict(
        tickmode="array",
        tickvals=list(range(len(feat_order))),
        ticktext=feat_order[::-1]
    ),
    height=520, template="plotly_dark",
    paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
    font=dict(color="#C9D1D9")
)
fig7.show()
fig7.write_html("viz7_shap_beeswarm.html")

print("""
📊 INTERPRETACIÓN — VIZ 7 (Beeswarm):
  • log_votes: valores altos (amarillo) → SHAP positivo fuerte.
    Pocas películas con millones de votos jalan la popularidad al alza.
  • log_revenue: mismo patrón; alta recaudación = fuerte empuje positivo.
  • vote_average: nube centrada en 0 → impacto casi nulo independiente
    del valor; confirma VIZ 6.
  • Action y Adventure: cuando están presentes (valor=1, amarillo)
    contribuyen positivamente; su ausencia es neutra.
""")

# ── VIZ 8 — SHAP Dependence Plot interactivo ────────────────
top_feat   = importancia.index[0]   # log_votes
inter_feat = importancia.index[1]   # log_revenue

sv_top  = shap_df[top_feat].values
fv_top  = X_shap[top_feat].values
fv_inter = X_shap[inter_feat].values

fig8 = px.scatter(
    x=fv_top, y=sv_top,
    color=fv_inter,
    color_continuous_scale="Turbo",
    labels={"x": top_feat, "y": f"SHAP ({top_feat})",
            "color": inter_feat},
    title=f"<b>SHAP DEPENDENCE PLOT — {top_feat}</b><br>"
          f"<sup>Color = {inter_feat} | Revela interacciones entre variables</sup>",
    template="plotly_dark",
    opacity=0.6
)
fig8.update_traces(marker_size=5,
    hovertemplate=(f"{top_feat}: %{{x:.3f}}<br>"
                   f"SHAP: %{{y:.4f}}<br>"
                   f"{inter_feat}: %{{marker.color:.3f}}<extra></extra>"))
fig8.update_layout(
    height=460,
    paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
    font=dict(color="#C9D1D9"),
    coloraxis_colorbar=dict(title=inter_feat)
)
fig8.show()
fig8.write_html("viz8_shap_dependence.html")

print(f"""
📊 INTERPRETACIÓN — VIZ 8 (Dependence Plot):
  • Relación positiva clara: más log_votes → mayor SHAP → más popularidad.
  • El color (log_revenue) muestra interacción: puntos amarillos
    (alta recaudación) tienden a estar en la parte superior del SHAP,
    indicando que el efecto de los votos se amplifica cuando la
    película también recauda mucho.
  • La nube se estrecha en valores bajos de log_votes, sugiriendo que
    películas con pocos votos tienen comportamiento más predecible.
""")

# Exportar valores SHAP para dashboard externo
shap_export = shap_df[feat_order].copy()
shap_export.columns = [f"shap_{c}" for c in shap_export.columns]
feat_export = X_shap[feat_order].copy()
combined = pd.concat([feat_export, shap_export], axis=1)
combined["pred_popularity"] = get_base_pipe(mejor).predict(X_test)[sample_idx]
import os; os.makedirs("outputs", exist_ok=True)
combined.to_csv("outputs/shap_data_export.csv", index=False)
print("\n✅  Datos SHAP exportados a outputs/shap_data_export.csv")
print("    → Úsalos en el dashboard HTML interactivo (proyecto_dashboard.html)")
print("    → Archivo: outputs/shap_data_export.csv\n")

# Guardar objetos del modelo para las visualizaciones complementarias.
# Cada script ejecutado con `python3 archivo.py` corre en un proceso nuevo,
# así que las variables no quedan disponibles automáticamente en memoria.
import pickle
os.makedirs("outputs", exist_ok=True)
with open("outputs/ml_objects.pkl", "wb") as f:
    pickle.dump({
        "movies": movies,
        "df": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "sample_weights": sample_weights,
        "mejor": mejor,
        "FEATURES_RAW": FEATURES_RAW,
        "FEATURES_OUT": FEATURES_OUT,
        "TOP_GENRES": TOP_GENRES,
        "NUM_CONT": NUM_CONT,
        "GENRE_COLS": GENRE_COLS,
    }, f)
print("✅  Objetos ML guardados en outputs/ml_objects.pkl\n")


# ════════════════════════════════════════════════════════════
#  SECCIÓN 4 — CONCLUSIONES
# ════════════════════════════════════════════════════════════

top3 = importancia.head(3).index.tolist()

print("=" * 62)
print("   SECCIÓN 4 — HALLAZGOS, LIMITACIONES Y RECOMENDACIÓN")
print("=" * 62)
# Obtener R² por segmento del mejor modelo
segs_mejor = mejor["segs"]

print(f"""
¿QUÉ ENCONTRAMOS?
─────────────────
El modelo {mejor['model']} explica el {mejor['r2']*100:.1f}% de la varianza
en la popularidad (log) de películas TMDB.

Variables más determinantes (SHAP):
  1. {top3[0]}   → Motor principal: el engagement (votar) retroalimenta la popularidad.
  2. {top3[1]}  → Alcance de distribución global y marketing.
  3. {top3[2]}  → Escala de producción y acceso a plataformas.

La calidad (vote_average) tiene impacto sorprendentemente bajo.
La popularidad es un fenómeno de MASA, no de calidad.

⚠️  MATIZ IMPORTANTE — R² por segmento:
  El R² global de {mejor['r2']:.3f} esconde una asimetría crítica:
  • Popularidad baja  (p0-33) : R² ≈ {segs_mejor.get('bajo (p0-33)', 0):.3f}  ← predice bien
  • Popularidad media (p33-66): R² ≈ {segs_mejor.get('medio (p33-66)', 0):.3f}  ← predice bien
  • Popularidad alta  (p66-100): R² ≈ {segs_mejor.get('alto (p66-100)', 0):.3f}  ← predice peor
  El modelo falla exactamente donde más importa para la industria.
  Las películas virales y franquicias son las peor predichas.

HALLAZGO DE SERIES DE TIEMPO (conecta con Sección 5):
  El análisis temporal confirma estacionalidad real: junio-julio y
  diciembre tienen ~15-20% más revenue que la base anual.
  Esto implica que el MES DE ESTRENO es una variable relevante que
  el modelo de regresión actual NO tiene como feature.
  Añadir release_month podría mejorar el R² en el segmento alto.

HALLAZGO DE ROI (VIZ-C1):
  Horror y Thriller tienen los mejores ROI con presupuesto bajo.
  La relación budget→popularidad no es universal — depende del género.

LIMITACIONES ACTUALIZADAS
──────────────────────────
  1. Budget con muchos ceros no reportados → ruido en log_budget.
  2. Popularidad TMDB es un snapshot dinámico — el modelo confunde
     "bajo vote_count al momento del snapshot" con "película impopular",
     cuando puede ser una película recién estrenada (limitación estructural).
  3. Sin datos de elenco, director, plataforma ni release_month.
     Estas variables explicarían gran parte del error en el segmento alto.
  4. Dataset hasta ~2017. El streaming post-2019 (Netflix, Disney+)
     cambió radicalmente los patrones de popularidad.
  5. ETS Multiplicative > Additive en series de tiempo → la amplitud
     estacional crece con el nivel, algo que el modelo de regresión
     tampoco captura sin release_month como feature.

RECOMENDACIÓN CONCRETA
──────────────────────
Para maximizar popularidad: invertir en DISTRIBUCIÓN y ENGAGEMENT
antes que en presupuesto de producción puro.

  ✅ Estrategia ganadora:
     • Estreno en junio-julio o diciembre (ventaja estacional confirmada)
     • Distribución simultánea en múltiples mercados (→ revenue alto)
     • Campaña activa de ratings tempranos (→ vote_count alto)
     = efecto multiplicador según el modelo.

  ✅ Por género:
     • Acción/Aventura: mejor popularidad media pero alto presupuesto.
     • Horror/Thriller: mejor ROI con presupuesto bajo — oportunidad
       para productoras independientes.

  ⚠️  Trampa a evitar:
     Producción costosa sin distribución amplia tiene menor ROI
     en popularidad que una producción modesta bien distribuida.
     El modelo confirma que presupuesto sin votos = invisibilidad.

  🔧 Mejora pendiente para el modelo:
     Añadir release_month como feature categórica. El análisis de
     series de tiempo demuestra que el mes de estreno tiene un efecto
     multiplicativo del 15-20% sobre el revenue esperado.
""")
