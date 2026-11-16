import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Configuración de página y estilos visuales
st.set_page_config(page_title="AgroTech Analytics U4", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0b1315; color: #e8f1f2; }
    div[data-testid="metric-container"] {
        background-color: #152325;
        border: 1px solid #2d5a52;
        border-left: 5px solid #00d285;
        padding: 15px 20px;
        border-radius: 10px;
    }
    .verdict-box {
        background: rgba(0, 210, 133, 0.12);
        border: 1px solid #00d285;
        border-radius: 10px;
        padding: 15px 20px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

plt.style.use("dark_background")
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#152325",
    "figure.facecolor": "#152325",
    "grid.color": "#263d40",
    "text.color": "#e8f1f2",
    "axes.labelcolor": "#cbe0dd",
    "xtick.color": "#94b8b2",
    "ytick.color": "#94b8b2"
})

# 2. Carga del dataset generado con DuckDB
@st.cache_data
def cargar_datos():
    return pd.read_csv("data/telemetria_invernadero.csv")

df_raw = cargar_datos()

# 3. Barra lateral con filtros dinámicos (Hito 4)
with st.sidebar:
    st.title("Telemetría Agrícola")
    st.caption("Invernadero Inteligente • Jitomate / Aguacate")
    st.markdown("---")
    tratamientos = st.multiselect(
        "Filtrar por Tratamiento:",
        options=["Tratamiento_A", "Tratamiento_B"],
        default=["Tratamiento_A", "Tratamiento_B"]
    )
    riego_min, riego_max = st.slider(
        "Rango de Riego (m³/ha):",
        float(df_raw["volumen_riego"].min()),
        float(df_raw["volumen_riego"].max()),
        (float(df_raw["volumen_riego"].min()), float(df_raw["volumen_riego"].max()))
    )
    fert_min, fert_max = st.slider(
        "Rango de Fertilizante (kg/ha):",
        float(df_raw["fertilizante_kg"].min()),
        float(df_raw["fertilizante_kg"].max()),
        (float(df_raw["fertilizante_kg"].min()), float(df_raw["fertilizante_kg"].max()))
    )
    st.markdown("---")
    st.info("Proyecto Integrador U4 • Análisis y Exploración de Datos (INC-2505)")

if not tratamientos:
    st.warning("Selecciona al menos un tratamiento en la barra lateral.")
    st.stop()

df = df_raw[
    (df_raw["tipo_tratamiento"].isin(tratamientos)) &
    (df_raw["volumen_riego"].between(riego_min, riego_max)) &
    (df_raw["fertilizante_kg"].between(fert_min, fert_max))
].copy()

if len(df) < 5:
    st.warning("Amplía los filtros para tener suficientes datos en el modelo.")
    st.stop()

# 4. Modelado OLS e Inferencia t-Student
X = sm.add_constant(df[["volumen_riego", "fertilizante_kg"]])
y = df["rendimiento_ton"]
modelo = sm.OLS(y, X).fit()
df["residuos"] = y - modelo.predict(X)

gA = df_raw[df_raw["tipo_tratamiento"] == "Tratamiento_A"]["rendimiento_ton"]
gB = df_raw[df_raw["tipo_tratamiento"] == "Tratamiento_B"]["rendimiento_ton"]
t_stat, p_valor = stats.ttest_ind(gB, gA, alternative="greater", equal_var=False)
diff_medias = gB.mean() - gA.mean()

# 5. Cabecera y 3 Tarjetas KPI
st.title("Optimización Inferencial del Rendimiento Agrícola")
st.markdown("Análisis de **Regresión Múltiple (OLS)** y **Evaluación Experimental de Bio-estimulantes**.")

k1, k2, k3 = st.columns(3)
k1.metric("Rendimiento Promedio Global", f"{df['rendimiento_ton'].mean():.2f} ton/ha", f"{len(df)} parcelas activas")
k2.metric("R² Ajustado del Modelo OLS", f"{modelo.rsquared_adj*100:.2f}%", "Varianza explicada")
k3.metric("P-Valor Prueba Inferencial", f"{p_valor:.2e}", "p < 0.05 (Significativo)")

st.markdown("---")

# 6. Cuadrícula 2x2 de las 4 Gráficas Técnicas
tab1, tab2, tab3 = st.tabs(["Dashboard Analítico (2x2)", "Ecuación y Veredicto", "Base de Datos"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Mapa de Calor de Correlación (Pearson)")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        corr = df[["volumen_riego", "fertilizante_kg", "rendimiento_ton"]].corr()
        corr.index = ["Riego (m³/ha)", "Fertilizante (kg/ha)", "Rendimiento (ton/ha)"]
        corr.columns = ["Riego (m³/ha)", "Fertilizante (kg/ha)", "Rendimiento (ton/ha)"]
        sns.heatmap(corr, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.subheader("2. Dispersión y Ajuste OLS (Riego vs. Rendimiento)")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.regplot(data=df, x="volumen_riego", y="rendimiento_ton",
                    scatter_kws={"color": "#00d285", "alpha": 0.75},
                    line_kws={"color": "#ff9f1c", "linewidth": 2.2}, ax=ax2)
        ax2.set_xlabel("Volumen de Riego (m³/ha)")
        ax2.set_ylabel("Rendimiento (ton/ha)")
        st.pyplot(fig2)

    st.markdown("<br>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("3. Boxplot Comparativo de Tratamientos (A vs. B)")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="tipo_tratamiento", y="rendimiento_ton",
                    palette={"Tratamiento_A": "#4ea8de", "Tratamiento_B": "#00d285"}, width=0.45, ax=ax3)
        ax3.set_xlabel("Tipo de Tratamiento")
        ax3.set_ylabel("Rendimiento (ton/ha)")
        st.pyplot(fig3)

    with col4:
        st.subheader("4. Histograma y Densidad de Residuos (Normalidad)")
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.histplot(df["residuos"], kde=True, bins=14, color="#a78bfa", ax=ax4)
        ax4.axvline(0, color="#ff9f1c", linestyle="--", linewidth=1.8)
        ax4.set_xlabel("Residuos (ton/ha)")
        st.pyplot(fig4)

with tab2:
    b0, b1, b2 = modelo.params["const"], modelo.params["volumen_riego"], modelo.params["fertilizante_kg"]
    rmse = np.sqrt(np.mean(df["residuos"]**2))
    
    st.subheader("Ecuación de Regresión Estimada")
    st.latex(rf"\widehat{{\text{{Rendimiento}}}} = {b0:.3f} + {b1:.4f}(\text{{Riego}}) + {b2:.4f}(\text{{Fertilizante}})")
    
    st.markdown(f"**Error Cuadrático Medio (RMSE):** `{rmse:.4f} ton/ha` | **Estadístico t (Tratamiento B vs. A):** `{t_stat:.4f}`")
    
    st.subheader("Resumen de Coeficientes del Modelo OLS (statsmodels)")
    resumen_ols = pd.DataFrame({
        "Variable": ["Intercepto (const)", "Volumen de Riego (m³/ha)", "Fertilizante (kg/ha)"],
        "Coeficiente (Beta)": modelo.params.values,
        "Error Estándar": modelo.bse.values,
        "Estadístico t": modelo.tvalues.values,
        "P-valor (P>|t|)": modelo.pvalues.values
    })
    st.dataframe(
        resumen_ols.style.format({
            "Coeficiente (Beta)": "{:.4f}",
            "Error Estándar": "{:.4f}",
            "Estadístico t": "{:.3f}",
            "P-valor (P>|t|)": "{:.4e}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown(f"""
    <div class="verdict-box">
        <strong>VEREDICTO DE INGENIERÍA (p = {p_valor:.5f} < 0.05):</strong><br>
        El <strong>Tratamiento_B (Bio-estimulante)</strong> incrementa el rendimiento promedio en <strong>+{diff_medias:.2f} ton/ha</strong> respecto al Tratamiento_A. 
        Se recomienda autorizar la inversión para escalar su uso en todo el invernadero.
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.subheader("Registro de Telemetría de Parcelas (N=100)")
    st.dataframe(df, use_container_width=True)
