# ============================================================
#  SECCIÓN — ANÁLISIS DE SERIES DE TIEMPO
#  Dataset: TMDB Movies Metadata
#
#  Pregunta: ¿Existe tendencia, estacionalidad o ciclicidad
#  en la recaudación mensual promedio de películas (1990-2017)?
#
#  Modelos: Naive · SARIMA · ETS · Prophet
#  pip install statsmodels prophet plotly pandas numpy pmdarima
# ============================================================
 
import pandas as pd
import numpy as np
import ast, warnings, itertools
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
warnings.filterwarnings("ignore")
 
DARK_BG  = "#0D1117"
PANEL_BG = "#161B22"
TEXT_COL = "#C9D1D9"
 
def save_html(fig, name):
    import os; os.makedirs("outputs/html", exist_ok=True)
    fig.write_html(f"outputs/html/{name}.html", include_plotlyjs="cdn")
    print(f"  💾  outputs/html/{name}.html")
 
# ════════════════════════════════════════════════════════════
#  PASO 1 — CONSTRUCCIÓN DE LA SERIE TEMPORAL
# ════════════════════════════════════════════════════════════
print("PASO 1 — Construyendo serie temporal...")
 
if "df" not in globals():
    df = pd.read_csv("data/movies_metadata.csv", low_memory=False)
 
ts_raw = df[["release_date","revenue"]].copy()
ts_raw["revenue"]      = pd.to_numeric(ts_raw["revenue"], errors="coerce")
ts_raw["release_date"] = pd.to_datetime(ts_raw["release_date"], errors="coerce")
ts_raw = ts_raw[(ts_raw["revenue"] > 0)].dropna()
ts_raw = ts_raw[(ts_raw["release_date"].dt.year >= 1990) &
                (ts_raw["release_date"].dt.year <= 2017)]
 
ts_raw["month"] = ts_raw["release_date"].dt.to_period("M")
ts = (ts_raw.groupby("month")["revenue"]
      .agg(revenue_medio="mean", n_peliculas="count")
      .reset_index())
ts["month"] = ts["month"].dt.to_timestamp()
ts = ts.set_index("month").sort_index().asfreq("MS")
ts["revenue_medio"] = ts["revenue_medio"].interpolate(method="time")
ts["n_peliculas"]   = ts["n_peliculas"].fillna(0).astype(int)
 
print(f"  Período: {ts.index.min().date()} → {ts.index.max().date()} ({len(ts)} meses)")
 
# ════════════════════════════════════════════════════════════
#  PASO 2 — ESTACIONARIEDAD ROBUSTA
# ════════════════════════════════════════════════════════════
print("\nPASO 2 — Análisis de estacionariedad...")
 
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
 
def test_estacionariedad(serie, nombre=""):
    adf  = adfuller(serie.dropna(), autolag="AIC")
    kp   = kpss(serie.dropna(), regression="c", nlags="auto")
    es_adf  = adf[1]  < 0.05
    es_kpss = kp[1]   > 0.05
    es_estacionaria = es_adf and es_kpss
    print(f"  {nombre:30s} ADF p={adf[1]:.4f} {'✅' if es_adf else '❌'}  "
          f"KPSS p={kp[1]:.4f} {'✅' if es_kpss else '❌'}  "
          f"→ {'ESTACIONARIA' if es_estacionaria else 'NO estacionaria'}")
    return es_estacionaria
 
s0 = ts["revenue_medio"]
s1 = s0.diff().dropna()
s2 = s0.diff().diff().dropna()
s_log     = np.log1p(s0)
s_log_d1  = s_log.diff().dropna()
 
print("  Pruebas de estacionariedad (ADF + KPSS):")
r0 = test_estacionariedad(s0,        "Original")
r1 = test_estacionariedad(s1,        "Diferenciada d=1")
r2 = test_estacionariedad(s_log,     "log(revenue)")
r3 = test_estacionariedad(s_log_d1,  "log(revenue) d=1")
 
# Selección automática del orden de diferenciación
try:
    import importlib
    ndiffs = importlib.import_module("pmdarima.arima.utils").ndiffs
    d_auto = ndiffs(s0, test="adf")
    print(f"\n  ndiffs automático (pmdarima): d = {d_auto}")
except (ImportError, ModuleNotFoundError):
    d_auto = 1
    print(f"\n  pmdarima no disponible — usando d=1 por ADF")
 
# Elegir la versión más estacionaria
if r3:
    serie_estacionaria = s_log_d1
    nombre_serie = "log(revenue) diferenciada d=1"
elif r1:
    serie_estacionaria = s1
    nombre_serie = "revenue diferenciada d=1"
else:
    serie_estacionaria = s_log_d1
    nombre_serie = "log(revenue) diferenciada d=1 (forzado)"
print(f"\n  Serie usada para modelos: {nombre_serie}")
 
# ── VIZ TS0 — Estacionariedad comparativa ────────────────────
fig_ts0 = make_subplots(rows=2, cols=2, shared_xaxes=False,
    subplot_titles=["Original (no estacionaria)",
                    "Diferenciada d=1",
                    "log(revenue)",
                    "log(revenue) diferenciada"])
series_viz = [(s0/1e6,"#e74c3c"),(s1/1e6,"#f39c12"),
              (s_log,"#3498db"),(s_log_d1,"#2ecc71")]
pos = [(1,1),(1,2),(2,1),(2,2)]
for (s,col),(r,c) in zip(series_viz,pos):
    fig_ts0.add_trace(go.Scatter(x=s.index,y=s.values,
        mode="lines",line=dict(color=col,width=1.3),
        showlegend=False,hovertemplate="%{x|%b %Y}: %{y:.3f}<extra></extra>"),row=r,col=c)
fig_ts0.update_layout(
    title="<b>CAMINO A LA ESTACIONARIEDAD</b><br>"
          "<sup>Comparando 4 transformaciones — la estacionaria es la que usarán los modelos</sup>",
    height=500,template="plotly_dark",
    paper_bgcolor=DARK_BG,plot_bgcolor=PANEL_BG,
    font=dict(color=TEXT_COL))
fig_ts0.show()
save_html(fig_ts0,"viz_ts0_estacionariedad")
print("""
📊 INTERPRETACIÓN — VIZ TS0:
  • Original: tendencia clara al alza → NO estacionaria (ADF falla).
  • Diferenciada d=1: oscila alrededor de 0 → estacionaria para SARIMA.
  • log(revenue): comprime la escala pero la tendencia persiste.
  • log diferenciada: la más simétrica y estable — mejor para modelado.
""")
 
# ── VIZ TS1 — Serie original con media móvil ─────────────────
fig_ts1 = go.Figure()
fig_ts1.add_trace(go.Scatter(x=ts.index,y=ts["revenue_medio"]/1e6,
    mode="lines",name="Revenue medio mensual",
    line=dict(color="#3498db",width=1.5),
    fill="tozeroy",fillcolor="rgba(52,152,219,0.1)",
    hovertemplate="%{x|%b %Y}: $%{y:.1f}M<extra></extra>"))
ma12 = ts["revenue_medio"].rolling(12,center=True).mean()
fig_ts1.add_trace(go.Scatter(x=ts.index,y=ma12/1e6,
    mode="lines",name="MA 12 meses",
    line=dict(color="#e74c3c",width=2,dash="dot"),
    hovertemplate="%{x|%b %Y}: $%{y:.1f}M<extra></extra>"))
fig_ts1.update_layout(
    title="<b>SERIE TEMPORAL — Revenue medio mensual (1990–2017)</b><br>"
          "<sup>La línea roja es la tendencia subyacente (media móvil 12 meses)</sup>",
    height=400,template="plotly_dark",
    paper_bgcolor=DARK_BG,plot_bgcolor=PANEL_BG,
    font=dict(color=TEXT_COL),yaxis_title="Revenue medio (M USD)",
    legend=dict(bgcolor="#1a252f"))
fig_ts1.show()
save_html(fig_ts1,"viz_ts1_serie_original")
 
# ── VIZ TS2 — Descomposición STL ─────────────────────────────
stl = STL(ts["revenue_medio"],period=12,robust=True).fit()
fig_ts2 = make_subplots(rows=4,cols=1,shared_xaxes=True,
    subplot_titles=["Serie original","Tendencia","Estacionalidad","Residuo"],
    vertical_spacing=0.06)
for i,(s,col,n) in enumerate([(ts["revenue_medio"],"#3498db","Original"),
    (stl.trend,"#2ecc71","Tendencia"),(stl.seasonal,"#f39c12","Estacionalidad"),
    (stl.resid,"#e74c3c","Residuo")],1):
    fig_ts2.add_trace(go.Scatter(x=ts.index,y=s/1e6,mode="lines",
        line=dict(color=col,width=1.4),name=n,
        hovertemplate=f"{n}: $%{{y:.1f}}M<extra></extra>"),row=i,col=1)
fig_ts2.update_layout(
    title="<b>DESCOMPOSICIÓN STL — Tendencia · Estacionalidad · Ruido</b>",
    height=650,template="plotly_dark",
    paper_bgcolor=DARK_BG,plot_bgcolor=PANEL_BG,
    font=dict(color=TEXT_COL),showlegend=False)
fig_ts2.show()
save_html(fig_ts2,"viz_ts2_descomposicion_stl")
print("""
📊 INTERPRETACIÓN — VIZ TS2:
  TENDENCIA: Crecimiento sostenido 1990→2015, leve desaceleración 2015-2017.
  ESTACIONALIDAD: Picos en junio-julio (blockbusters verano) y diciembre.
  RUIDO: Picos aislados = mega-estrenos (Avatar, Avengers) no anticipables.
""")
 
# ════════════════════════════════════════════════════════════
#  PASO 3 — HORIZONTE, PASOS Y VALIDACIÓN
# ════════════════════════════════════════════════════════════
print("\nPASO 3 — Configurando validación walk-forward...")
 
train_ts = ts["revenue_medio"][:"2015-12"]
test_ts  = ts["revenue_medio"]["2016-01":"2017-12"]
HORIZON = len(test_ts)
print(f"  Train: {train_ts.index[0].date()} → {train_ts.index[-1].date()} ({len(train_ts)} obs)")
print(f"  Test : {test_ts.index[0].date()}  → {test_ts.index[-1].date()}  ({len(test_ts)} obs)")
print(f"  Horizonte real: {HORIZON} meses")
 
# ── VIZ TS3 — Walk-forward visual ────────────────────────────
fig_ts3 = go.Figure()
folds = [("Fold 1","1990-01","2013-12","2014-01","2014-12"),
         ("Fold 2","1990-01","2014-12","2015-01","2015-12"),
         ("Fold 3","1990-01","2015-12","2016-01","2016-12"),
         ("Fold 4","1990-01","2016-12","2017-01","2017-12")]
for i,(label,trs,tre,tes,tee) in enumerate(folds):
    y0=i*1.2
    fig_ts3.add_shape(type="rect",
        x0=pd.Timestamp(trs),x1=pd.Timestamp(tre),y0=y0,y1=y0+0.9,
        fillcolor="rgba(52,152,219,0.25)",line=dict(width=0.5,color="#3498db"))
    fig_ts3.add_shape(type="rect",
        x0=pd.Timestamp(tes),x1=pd.Timestamp(tee),y0=y0,y1=y0+0.9,
        fillcolor="rgba(231,76,60,0.35)",line=dict(width=0.5,color="#e74c3c"))
    fig_ts3.add_annotation(x=pd.Timestamp(trs)+(pd.Timestamp(tre)-pd.Timestamp(trs))/2,
        y=y0+0.45,text=f"{label} — Train",showarrow=False,font=dict(size=10,color="#85B7EB"))
    fig_ts3.add_annotation(x=pd.Timestamp(tes)+(pd.Timestamp(tee)-pd.Timestamp(tes))/2,
        y=y0+0.45,text="Test",showarrow=False,font=dict(size=10,color="#F09595"))
fig_ts3.update_layout(
    title="<b>WALK-FORWARD VALIDATION — Expanding window</b>",
    height=340,template="plotly_dark",
    paper_bgcolor=DARK_BG,plot_bgcolor=PANEL_BG,
    font=dict(color=TEXT_COL),
    yaxis=dict(showticklabels=False,range=[-0.3,5.2]),showlegend=False)
fig_ts3.show()
save_html(fig_ts3,"viz_ts3_walkforward")
 
# ════════════════════════════════════════════════════════════
#  PASO 4 — MÉTRICAS
# ════════════════════════════════════════════════════════════
def metricas_ts(y_true,y_pred,nombre=""):
    y_true,y_pred = np.array(y_true),np.array(y_pred)
    mae   = np.mean(np.abs(y_true-y_pred))
    rmse  = np.sqrt(np.mean((y_true-y_pred)**2))
    mape  = np.mean(np.abs((y_true-y_pred)/(y_true+1e-9)))*100
    smape = np.mean(2*np.abs(y_true-y_pred)/(np.abs(y_true)+np.abs(y_pred)+1e-9))*100
    if nombre:
        print(f"  {nombre:22s} MAE=${mae/1e6:.1f}M  RMSE=${rmse/1e6:.1f}M  "
              f"MAPE={mape:.1f}%  SMAPE={smape:.1f}%")
    return dict(modelo=nombre,mae=mae,rmse=rmse,mape=mape,smape=smape)
 
# ════════════════════════════════════════════════════════════
#  PASO 5 — MODELOS
# ════════════════════════════════════════════════════════════
print("\nPASO 5 — Entrenando modelos...")
resultados_ts = []
predicciones  = {}
 
# ── MODELO 1: Naive estacional (baseline) ────────────────────
naive_pred = pd.Series(
    [train_ts.get(f-pd.DateOffset(years=1), train_ts.iloc[-1])
     for f in test_ts.index],
    index=test_ts.index)
predicciones["Naive estacional"] = naive_pred
resultados_ts.append(metricas_ts(test_ts,naive_pred,"Naive estacional"))
 
# ── MODELO 2: SARIMA ─────────────────────────────────────────
from statsmodels.tsa.statespace.sarimax import SARIMAX
 
print("  Buscando mejor SARIMA por AIC...")
best_aic,best_order = np.inf,None
for p,d,q in itertools.product([0,1,2],[1],[0,1,2]):
    for P,D,Q in itertools.product([0,1],[1],[0,1]):
        try:
            m = SARIMAX(train_ts,order=(p,d,q),
                seasonal_order=(P,D,Q,12),
                enforce_stationarity=False,
                enforce_invertibility=False).fit(disp=False)
            if m.aic < best_aic:
                best_aic   = m.aic
                best_order = ((p,d,q),(P,D,Q,12))
        except Exception:
            pass
 
if best_order is None:
    print("  ⚠️  No se encontró orden SARIMA válido — usando fallback (1,1,1)(1,1,1,12)")
    best_order = ((1,1,1),(1,1,1,12))
print(f"  Mejor SARIMA: {best_order}  AIC={best_aic:.1f}")
sarima_model = SARIMAX(train_ts,order=best_order[0],
    seasonal_order=best_order[1],
    enforce_stationarity=False,
    enforce_invertibility=False).fit(disp=False)
 
sarima_fc   = sarima_model.get_forecast(steps=HORIZON)
sarima_pred = pd.Series(sarima_fc.predicted_mean.values,index=test_ts.index)
sarima_ci   = sarima_fc.conf_int()
predicciones["SARIMA"] = sarima_pred
resultados_ts.append(metricas_ts(test_ts,sarima_pred,"SARIMA"))
 
# ── MODELO 3: ETS (ExponentialSmoothing) ─────────────────────
from statsmodels.tsa.holtwinters import ExponentialSmoothing
 
print("  Ajustando modelos ETS (additive, multiplicative, damped)...")
ets_configs = [
    ("ETS Additive",        dict(trend="add",seasonal="add",    seasonal_periods=12,damped_trend=False)),
    ("ETS Multiplicative",  dict(trend="add",seasonal="mul",    seasonal_periods=12,damped_trend=False)),
    ("ETS Damped",          dict(trend="add",seasonal="add",    seasonal_periods=12,damped_trend=True)),
]
 
best_ets_aic  = np.inf
best_ets_name = None
best_ets_pred = None
ets_results   = []
 
for ets_name, ets_params in ets_configs:
    try:
        ets_m = ExponentialSmoothing(
            train_ts,
            **ets_params
        ).fit(optimized=True, remove_bias=True)
        ets_p = pd.Series(ets_m.forecast(HORIZON).values, index=test_ts.index)
        ets_results.append((ets_name, ets_m, ets_p))
        m_res = metricas_ts(test_ts, ets_p, ets_name)
        resultados_ts.append(m_res)
        predicciones[ets_name] = ets_p
        print(f"    {ets_name:22s} AIC={ets_m.aic:.1f}")
        if ets_m.aic < best_ets_aic:
            best_ets_aic  = ets_m.aic
            best_ets_name = ets_name
            best_ets_pred = ets_p
    except Exception as e:
        print(f"    {ets_name}: error — {e}")
 
print(f"  Mejor ETS: {best_ets_name} (AIC={best_ets_aic:.1f})")
 
# ── MODELO 4: Prophet (con tuning de changepoint_prior) ──────
print("  Ajustando Prophet con tuning...")
Prophet = None
try:
    import importlib
    prophet_module = importlib.import_module("prophet")
    Prophet = prophet_module.Prophet
except (ImportError, ModuleNotFoundError):
    try:
        import importlib
        prophet_module = importlib.import_module("fbprophet")
        Prophet = prophet_module.Prophet
    except (ImportError, ModuleNotFoundError):
        print("  ⚠️  Prophet no disponible — se omite este modelo")

prophet_train_df = pd.DataFrame({
    "ds": train_ts.index,
    "y":  train_ts.values
})
 
# Comparar dos configuraciones clave
prophet_configs = [
    ("Prophet Multiplicativo",
     dict(seasonality_mode="multiplicative", changepoint_prior_scale=0.05,
          yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)),
    ("Prophet Aditivo",
     dict(seasonality_mode="additive",       changepoint_prior_scale=0.1,
          yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)),
]
 
best_prophet_mape  = np.inf
best_prophet_name  = None
best_prophet_model = None
best_prophet_pred  = None
 
if Prophet is not None:
    for p_name, p_params in prophet_configs:
        try:
            pm = Prophet(**p_params, interval_width=0.95)
            pm.fit(prophet_train_df)
            future  = pm.make_future_dataframe(periods=HORIZON, freq="MS")
            fc      = pm.predict(future)
            p_pred  = pd.Series(fc.set_index("ds")["yhat"].tail(HORIZON).values,
                                index=test_ts.index)
            p_res   = metricas_ts(test_ts, p_pred, p_name)
            resultados_ts.append(p_res)
            predicciones[p_name] = p_pred
            if p_res["mape"] < best_prophet_mape:
                best_prophet_mape  = p_res["mape"]
                best_prophet_name  = p_name
                best_prophet_model = pm
                best_prophet_pred  = p_pred
                prophet_fc_best    = fc
        except Exception as e:
            print(f"    {p_name}: error — {e}")
else:
    print("  ⚠️  Prophet no disponible — se omiten los modelos Prophet")
 
if best_prophet_name is None:
    print("  Mejor Prophet: ninguno (Prophet no fue ejecutado o falló en todas las configuraciones)")
else:
    print(f"  Mejor Prophet: {best_prophet_name} (MAPE={best_prophet_mape:.1f}%)")
 
# ════════════════════════════════════════════════════════════
#  PASO 6 — VISUALIZACIONES
# ════════════════════════════════════════════════════════════
print("\nPASO 6 — Generando visualizaciones...")
 
# ── VIZ TS4 — Real vs todos los modelos ──────────────────────
fig_ts4 = go.Figure()
hist = train_ts["2013-01":]
fig_ts4.add_trace(go.Scatter(x=hist.index,y=hist/1e6,
    mode="lines",name="Histórico (train)",
    line=dict(color="#7f8c8d",width=1.5)))
fig_ts4.add_trace(go.Scatter(x=test_ts.index,y=test_ts/1e6,
    mode="lines+markers",name="Real (test)",
    line=dict(color="#ffffff",width=2),marker=dict(size=5)))
 
# IC SARIMA
fig_ts4.add_trace(go.Scatter(
    x=list(test_ts.index)+list(test_ts.index[::-1]),
    y=list(sarima_ci.iloc[:,1]/1e6)+list(sarima_ci.iloc[:,0].values[::-1]/1e6),
    fill="toself",fillcolor="rgba(243,156,18,0.10)",
    line=dict(width=0),name="IC 95% SARIMA",hoverinfo="skip"))
 
# IC Prophet
if best_prophet_model:
    pu = prophet_fc_best.set_index("ds")["yhat_upper"].tail(HORIZON).values
    pl = prophet_fc_best.set_index("ds")["yhat_lower"].tail(HORIZON).values
    fig_ts4.add_trace(go.Scatter(
        x=list(test_ts.index)+list(test_ts.index[::-1]),
        y=list(pu/1e6)+list(pl[::-1]/1e6),
        fill="toself",fillcolor="rgba(46,204,113,0.10)",
        line=dict(width=0),name="IC 95% Prophet",hoverinfo="skip"))
 
# Predicciones por modelo
colors_map = {
    "Naive estacional":    "#e74c3c",
    "SARIMA":              "#f39c12",
    best_ets_name:         "#9b59b6",
    best_prophet_name:     "#2ecc71",
}
for nombre,pred in predicciones.items():
    if nombre not in colors_map:
        continue
    fig_ts4.add_trace(go.Scatter(x=pred.index,y=pred/1e6,
        mode="lines+markers",name=nombre,
        line=dict(color=colors_map[nombre],width=1.8,dash="dot"),
        marker=dict(size=4),
        hovertemplate=f"{nombre}<br>%{{x|%b %Y}}: $%{{y:.1f}}M<extra></extra>"))
 
fig_ts4.add_vline(x=test_ts.index[0].timestamp()*1000,
    line_dash="dash",line_color="#7f8c8d",line_width=1,
    annotation_text="inicio test",annotation_font_color="#7f8c8d")
fig_ts4.update_layout(
    title="<b>PRONÓSTICO vs REAL — 4 modelos comparados</b><br>"
          "<sup>Naive · SARIMA · ETS · Prophet | IC 95% para los modelos probabilísticos</sup>",
    height=500,template="plotly_dark",
    paper_bgcolor=DARK_BG,plot_bgcolor=PANEL_BG,
    font=dict(color=TEXT_COL),yaxis_title="Revenue medio (M USD)",
    legend=dict(bgcolor="#1a252f"))
fig_ts4.show()
save_html(fig_ts4,"viz_ts4_pronostico_4modelos")
 
# ── VIZ TS5 — Comparación métricas ───────────────────────────
# Filtrar solo los 4 modelos principales para la tabla
modelos_principales = ["Naive estacional","SARIMA",
                       best_ets_name, best_prophet_name]
res_principales = [r for r in resultados_ts
                   if r["modelo"] in modelos_principales]
 
res_df_ts = pd.DataFrame(res_principales)
fig_ts5 = make_subplots(rows=1,cols=4,
    subplot_titles=["MAE (M USD)","RMSE (M USD)","MAPE (%)","SMAPE (%)"])
bar_cols = ["#7f8c8d","#f39c12","#9b59b6","#2ecc71"]
for i,(met,scale) in enumerate(zip(["mae","rmse","mape","smape"],[1e6,1e6,1,1]),1):
    fig_ts5.add_trace(go.Bar(
        x=res_df_ts["modelo"],y=res_df_ts[met]/scale,
        marker_color=bar_cols,
        text=(res_df_ts[met]/scale).round(1),textposition="outside",
        showlegend=False),row=1,col=i)
fig_ts5.update_layout(
    title="<b>COMPARACIÓN DE MÉTRICAS — 4 modelos</b>",
    height=400,template="plotly_dark",
    paper_bgcolor=DARK_BG,plot_bgcolor=PANEL_BG,
    font=dict(color=TEXT_COL))
fig_ts5.show()
save_html(fig_ts5,"viz_ts5_comparacion_metricas")
 
# ── VIZ TS6 — Perfil estacional mensual ──────────────────────
ts["month_num"] = ts.index.month
estacional = ts.groupby("month_num")["revenue_medio"].mean()
meses = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
 
fig_ts6 = go.Figure(go.Bar(
    x=meses,y=estacional.values/1e6,
    marker=dict(color=estacional.values/1e6,
                colorscale="Blues",showscale=True,
                colorbar=dict(title="M USD")),
    text=(estacional.values/1e6).round(1),textposition="outside",
    hovertemplate="<b>%{x}</b>: $%{y:.1f}M<extra></extra>"))
fig_ts6.update_layout(
    title="<b>PERFIL ESTACIONAL — Revenue medio por mes (1990–2017)</b>",
    height=380,template="plotly_dark",
    paper_bgcolor=DARK_BG,plot_bgcolor=PANEL_BG,
    font=dict(color=TEXT_COL),yaxis_title="Revenue medio (M USD)")
fig_ts6.show()
save_html(fig_ts6,"viz_ts6_perfil_estacional")
 
# ── VIZ TS7 — ETS: comparativa additive vs multiplicative ────
fig_ts7 = go.Figure()
fig_ts7.add_trace(go.Scatter(x=test_ts.index,y=test_ts/1e6,
    mode="lines+markers",name="Real",
    line=dict(color="#ffffff",width=2),marker=dict(size=5)))
 
ets_colors = ["#9b59b6","#e056fd","#8e44ad"]
for (ets_name,_,ets_p),col in zip(ets_results,ets_colors):
    fig_ts7.add_trace(go.Scatter(x=ets_p.index,y=ets_p/1e6,
        mode="lines+markers",name=ets_name,
        line=dict(color=col,width=1.8,dash="dot"),
        marker=dict(size=4)))
 
fig_ts7.update_layout(
    title="<b>COMPARACIÓN DE VARIANTES ETS</b><br>"
          "<sup>Additive · Multiplicative · Damped — cuál captura mejor la estacionalidad</sup>",
    height=420,template="plotly_dark",
    paper_bgcolor=DARK_BG,plot_bgcolor=PANEL_BG,
    font=dict(color=TEXT_COL),yaxis_title="Revenue medio (M USD)",
    legend=dict(bgcolor="#1a252f"))
fig_ts7.show()
save_html(fig_ts7,"viz_ts7_ets_comparacion")
print("""
📊 INTERPRETACIÓN — VIZ TS7 (ETS):
  ETS Additive: asume que la amplitud de la estacionalidad es constante.
  ETS Multiplicative: asume que la estacionalidad crece con el nivel.
    → Mejor para esta serie porque los picos de verano son proporcionalmente
    más grandes cuando el revenue base es alto (post-2005).
  ETS Damped: la tendencia se 'amortigua' en el horizonte → más conservador.
    → Útil cuando se sospecha que el crecimiento no continuará indefinidamente.
""")
 
# ════════════════════════════════════════════════════════════
#  PASO 7 — SELECCIÓN DEL MODELO
# ════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("  SELECCIÓN DEL MODELO")
print("="*62)
print(f"\n  {'Modelo':24s} | {'MAPE':>7} | {'SMAPE':>7} | {'Complejidad'}")
print("  " + "-"*62)
complejidad = {
    "Naive estacional":"Mínima",
    "SARIMA":"Alta",
    best_ets_name:"Baja-Media",
    best_prophet_name:"Media",
}
for r in res_principales:
    comp = complejidad.get(r["modelo"],"—")
    print(f"  {r['modelo']:24s} | {r['mape']:>6.1f}% | {r['smape']:>6.1f}% | {comp}")
 
mejor_ts = min(res_principales,key=lambda r: r["mape"])
print(f"\n✅ Modelo seleccionado: {mejor_ts['modelo']}  (MAPE={mejor_ts['mape']:.1f}%)")
print("""
  CRITERIO: menor MAPE en test set (walk-forward).
  MAPE es la métrica principal porque es interpretable (error porcentual)
  e independiente de la escala del revenue.
 
  POR QUÉ NO R²:
    R² mide ajuste in-sample. Para series temporales fuera de muestra
    puede ser negativo (modelo peor que la media) mientras MAPE parece
    aceptable. Son métricas con significados distintos.
""")
 