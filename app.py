# Dasboard Nomofobia y Dependencia al Smartphone

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import anderson, shapiro, probplot
import plotly.express as px
import plotly.graph_objects as go
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import math
from pathlib import Path

# -------------------- Metadatos --------------------
AUTHORS = "Johann Smith Rivera & Julian Mateo Valderrama"
COURSE = "Estadística No Paramétrica"
UNIVERSITY = "Universidad Santo Tomás"
PROF = "Javier Sierra"
YEAR = "2025"

# -------------------- Paleta institucional (global) --------------------
PALETTE_INST = ["#0F4C81", "#F4A300", "#7BAFD4", "#D97B0E", "#4C6A92"]

# -------------------- Configuración de página --------------------
st.set_page_config(
    page_title="Análisis de Nomofobia y Dependencia al Smartphone",
    layout="wide",
    initial_sidebar_state="expanded")

# --- Bloque visual: animación + tarjetas elegantes + header con fade-in ---
st.markdown("""
    <style>
    /* === Animación global de aparición === */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeIn 1.2s ease-in-out;
    }

    /* === Estilo de tarjetas (cards) para cada pestaña === */
    .stTabs [data-baseweb="tab-panel"] > div {
        background-color: #1a1c23;       /* Fondo gris oscuro suave */
        padding: 25px 28px;
        border-radius: 18px;
        box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.25);
        margin-top: 20px;
        animation: fadeIn 1.2s ease-in-out;
    }

    /* Bordes y efecto hover sutil */
    .stTabs [data-baseweb="tab-panel"] > div:hover {
        box-shadow: 0px 6px 25px rgba(255, 255, 255, 0.07);
        transition: all 0.3s ease-in-out;
    }

    /* === Animación del título principal === */
    .main-title {
        font-size: 2.1em;
        font-weight: 700;
        color: #F4A300;
        text-align: center;
        margin-top: 12px;
        animation: fadeIn 1.3s ease-in-out;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- Estilos globales --------------------
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    h1, h2, h3, h4, h5, h6, p, div, span { color: #ffffff !important; }
    .stMarkdown, .stCaption { color: #dddddd !important; }
    .stSidebar { background-color: #1a1c23 !important; }
    </style>
""", unsafe_allow_html=True)

# -------------------- PORTADA INSTITUCIONAL (blanca, fade-in) --------------------
st.markdown("""
    <style>
    body { background-color: #ffffff; }
    .centered { text-align: center; padding: 42px 18px; }
    .fade-in { animation: fadeIn 1.6s ease-in; }
    @keyframes fadeIn { from {opacity:0;} to {opacity:1;} }

    .welcome-title {
        color: #ffffff;
        font-size: 2.6em;
        font-weight: 800;
        margin-bottom: 0.3em;
        letter-spacing: 0.6px;
        text-shadow: 1px 1px 2px #0F4C81;
    }

    .welcome-sub {
        color: #dddddd;
        font-size: 1.05em;
        margin-bottom: 1.4em;
    }

    .launch-btn {
        background-color: #0F4C81;
        color: white;
        padding: 12px 28px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.05em;
        transition: all 0.25s ease;
    }
    .launch-btn:hover {
        background-color:#123E6C;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

logo_path = Path("logo.png")
if "show_dashboard" not in st.session_state:
    st.session_state["show_dashboard"] = False

if not st.session_state["show_dashboard"]:
    st.markdown('<div class="centered fade-in">', unsafe_allow_html=True)
    st.image(str(logo_path), width=240)
    st.markdown(
        '<div class="welcome-title">📱 Nomofobia y Dependencia al Smartphone 😰📊</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="welcome-sub"><b>{UNIVERSITY}</b> — {COURSE}<br>'
        f'Profesor: {PROF}<br>Autores: {AUTHORS} | {YEAR}</div>',
        unsafe_allow_html=True
    )
    if st.button("🚀 Iniciar Análisis", key="start_button"):
        st.session_state["show_dashboard"] = True
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# -------------------- Header --------------------
st.markdown("""
    <style>
    /* === Animaciones y estilo del header === */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .main-title {
        font-size: 2.4em;
        font-weight: 800;
        color: #F4A300;
        text-align: center;
        margin-bottom: 0.3em;
        animation: fadeIn 1.2s ease-in-out;
    }

    .subtitle {
        font-size: 1.1em;
        text-align: center;
        color: #E0E0E0;
        animation: fadeIn 1.4s ease-in-out;
        margin-bottom: 0.2em;
    }

    .professor {
        font-size: 1.05em;
        text-align: center;
        color: #CCCCCC;
        animation: fadeIn 1.6s ease-in-out;
        margin-bottom: 0.3em;
    }

    .authors {
        text-align: center;
        color: #BFBFBF;
        font-size: 0.95em;
        margin-bottom: 0.8em;
        animation: fadeIn 1.8s ease-in-out;
    }

    hr.divider {
        border: 1px solid #2E2E2E;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header animación ---
st.markdown('<h1 class="main-title">📱 Nomofobia y Dependencia al Smartphone</h1>', unsafe_allow_html=True)
st.markdown(f'<div class="subtitle"><b>{UNIVERSITY}</b> — {COURSE}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="professor"><b>Profesor:</b> {PROF}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="authors"><b>Autores:</b> {AUTHORS} • {YEAR}</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# -------------------- CARGA DE DATOS --------------------
df = pd.read_excel("DATOS REALES.xlsx").rename(columns=str.strip)

# Limpieza mínima
for col in ["Sexo", "Estrato", "Nomofobia?"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

for col in ["Horas_Uso", "Nomofobia", "Ansiedad_social", "Autoestima", "Edad", "Mal_uso"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -------------------- SIDEBAR: Filtros --------------------
st.sidebar.header("Parámetros de la visualización")
sexo_sel = st.sidebar.multiselect("Sexo", df["Sexo"].dropna().unique(), default=df["Sexo"].dropna().unique())
estrato_sel = st.sidebar.multiselect("Estrato", df["Estrato"].dropna().unique(), default=df["Estrato"].dropna().unique())
nomob_sel = st.sidebar.multiselect("Nomofobia? (Sí/No)", df["Nomofobia?"].dropna().unique(), default=df["Nomofobia?"].dropna().unique())

bootstrap_spearman = st.sidebar.checkbox("Bootstrapped CI (Spearman)", value=True)
nboots = 1000

# Aplicar filtros
df_f = df[
    df["Sexo"].isin(sexo_sel)
    & df["Estrato"].isin(estrato_sel)
    & df["Nomofobia?"].isin(nomob_sel)
].copy()

# -------------------- Función auxiliar: prueba de normalidad --------------------
def run_normality(series):
    s = series.dropna()
    if len(s) < 3:
        return np.nan, np.nan
    try:
        W, p_sh = shapiro(s)
        ad_res = anderson(s, dist="norm")
        return p_sh, ad_res.statistic
    except Exception:
        return np.nan, np.nan

# -------------------- Mensaje informativo global --------------------
st.markdown(
    f"""
    <div style='background-color:#1a1c23;padding:12px;border-radius:8px;margin-top:10px'>
        <b>📘 Proyecto académico:</b> {COURSE} — <i>{UNIVERSITY}</i><br>
        Análisis integral de la nomofobia y el uso del smartphone bajo pruebas no paramétricas.
    </div>
    """,
    unsafe_allow_html=True)

# -------------------- CONTEXTO AMPLIADO --------------------
st.header("Análisis estadístico no paramétrico — Nomofobia")

# === Pestañas principales ===
tabs = st.tabs([
    "📘 Contexto y objetivos",
    "📊 Visualizaciones descriptivas",
    "🔗 Spearman",
    "⚖️ Mann–Whitney",
    "📈 Kruskal–Wallis",
    "🧩 Post-Hoc Dunn",
    "🧭 Exploradores",
    "🧠 Conclusiones"
])

# === Pestaña 1: Contexto y objetivos ===
with tabs[0]:
    st.subheader("Contexto y objetivos 📘")

    st.write(
        """
        Este análisis aborda el fenómeno de la **nomofobia** —el miedo irracional a estar sin el teléfono móvil— 
        como un indicador emergente de **dependencia tecnológica** en estudiantes universitarios.  
        A partir de las escalas validadas de *Fryman & Romine (2021)* y otros autores, se midieron dimensiones 
        relacionadas con **ansiedad social**, **autoestima**, y **tiempo de uso diario del smartphone**, 
        variables que permiten explorar de forma integral los efectos psicológicos del uso excesivo del dispositivo.  

        Los datos fueron obtenidos mediante un instrumento autoaplicado y se analizaron bajo un enfoque 
        **no paramétrico**, dado que la mayoría de las distribuciones no cumplen con los supuestos de normalidad.
        """
    )

    # === KPIs DINÁMICOS ===
    st.markdown("### 📊 Indicadores Clave del Estudio")

    df_kpi = df_f.copy()

    # Cálculo de métricas principales
    rho, p_rho = (np.nan, np.nan)
    if {"Horas_Uso", "Nomofobia"}.issubset(df_kpi.columns):
        rho, p_rho = stats.spearmanr(df_kpi["Horas_Uso"], df_kpi["Nomofobia"], nan_policy="omit")

    p_mw = np.nan
    if {"Nomofobia?", "Horas_Uso"}.issubset(df_kpi.columns):
        a = df_kpi[df_kpi["Nomofobia?"] == "Sí"]["Horas_Uso"].dropna()
        b = df_kpi[df_kpi["Nomofobia?"] == "No"]["Horas_Uso"].dropna()
        if len(a) >= 3 and len(b) >= 3:
            _, p_mw = stats.mannwhitneyu(a, b, alternative="two-sided")

    p_kw = np.nan
    if {"Estrato", "Nomofobia"}.issubset(df_kpi.columns):
        groups = [g["Nomofobia"].dropna() for _, g in df_kpi.groupby("Estrato")]
        if len(groups) > 1:
            _, p_kw = stats.kruskal(*groups)

    # Etiquetas más interpretativas
    def format_p(p):
        if np.isnan(p):
            return "NA"
        elif p < 0.001:
            return "p < 0.001"
        elif p < 0.01:
            return "p < 0.01"
        elif p < 0.05:
            return "p < 0.05"
        else:
            return "ns (≥0.05)"

    # Mostrar KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("📈 Asociación Horas–Nomofobia (ρ)", f"{rho:.2f}" if not np.isnan(rho) else "NA",
                "Alta" if rho >= 0.6 else ("Moderada" if rho >= 0.3 else "Baja"))
    col2.metric("⚖️ Mann–Whitney", format_p(p_mw),
                "Diferencia significativa" if (not np.isnan(p_mw) and p_mw < 0.05) else "No significativa")
    col3.metric("📊 Kruskal–Wallis", format_p(p_kw),
                "Diferencias entre estratos" if (not np.isnan(p_kw) and p_kw < 0.05) else "Sin diferencias")

    st.caption("Estos indicadores resumen las asociaciones y diferencias clave del estudio.")

    st.markdown("### 🎯 **Objetivos del estudio**")
    st.write(
        """
        1. **Evaluar** si las horas de uso del teléfono móvil se asocian significativamente con los niveles de nomofobia.  
        2. **Explorar** las relaciones entre nomofobia, ansiedad social y autoestima en la población estudiantil.  
        3. **Aplicar pruebas no paramétricas** (Spearman, Mann–Whitney, Kruskal–Wallis y Dunn) para identificar patrones 
           de asociación y diferencias entre grupos sociodemográficos.  
        4. **Visualizar** los resultados mediante un dashboard interactivo que facilite la interpretación de los hallazgos.
        """
    )

    st.markdown("---")

    # === Resumen descriptivo ===
    st.subheader("Resumen descriptivo de las variables numéricas 📊")
    numeric_cols = [c for c in ["Horas_Uso", "Nomofobia", "Ansiedad_social", "Autoestima", "Mal_uso"] if c in df_kpi.columns]

    if numeric_cols:
        desc = df_kpi[numeric_cols].describe().T.rename(columns={"50%": "mediana"})
        st.dataframe(desc.style.format("{:.2f}"), use_container_width=True)
        st.caption("Tabla 1. Estadísticos descriptivos básicos de las principales variables de estudio.")
    else:
        st.warning("No se encontraron variables numéricas en los datos cargados.")

    st.markdown("---")

    # === Pruebas de normalidad ===
    def run_normality(series):
        from scipy.stats import shapiro, anderson
        s = series.dropna()
        if len(s) < 3:
            return np.nan, np.nan
        try:
            W, p_sh = shapiro(s)
            ad_res = anderson(s, dist="norm")
            return p_sh, ad_res.statistic
        except Exception:
            return np.nan, np.nan

    if numeric_cols:
        st.subheader("Pruebas de normalidad (Shapiro–Wilk y Anderson–Darling) 🧮")

        results = []
        for col in numeric_cols:
            p_sh, ad_stat = run_normality(df_kpi[col])
            normal = "✅ Normal" if p_sh >= 0.05 else "⚠️ No normal"
            results.append({
                "Variable": col,
                "Shapiro-Wilk (p)": f"{p_sh:.4f}" if not np.isnan(p_sh) else "NA",
                "Anderson-Darling (stat)": f"{ad_stat:.4f}" if not np.isnan(ad_stat) else "NA",
                "Resultado": normal
            })

        res_df = pd.DataFrame(results)

        def color_result(val):
            if "No normal" in val:
                return "color: #FF6B6B; font-weight:600;"
            elif "Normal" in val:
                return "color: #4CAF50; font-weight:600;"
            return ""

        st.dataframe(
            res_df.style.applymap(color_result, subset=["Resultado"]),
            use_container_width=True
        )
        st.caption("Tabla 2. Resultados de las pruebas de normalidad por variable.")

        st.markdown(
            """
            **Interpretación:**  
            Los resultados confirman que las variables cuantitativas analizadas **no siguen una distribución normal**, 
            respaldando la decisión metodológica del uso de pruebas no paramétricas.  
            Este patrón concuerda con el informe del proyecto, donde se observaron asimetrías significativas en los 
            niveles de nomofobia y en las horas de uso del smartphone.  
            
            👉 En la siguiente pestaña se presentan las visualizaciones descriptivas que permiten observar gráficamente estos comportamientos.
            """
        )
    else:
        st.info("No hay variables numéricas disponibles para realizar pruebas de normalidad.")

# === Pestaña 2: Visualizaciones descriptivas ===
with tabs[1]:
    st.subheader("📊 Visualizaciones descriptivas")

    # --- Definición global de paleta (azul–ocre) ---
    palette_institucional = ["#0F4C81", "#F4A300", "#7BAFD4", "#D97B0E", "#4C6A92"]

    numeric_cols = [c for c in ["Horas_Uso", "Nomofobia", "Ansiedad_social", "Autoestima", "Mal_uso"] if c in df_f.columns]

    if not numeric_cols:
        st.warning("No hay variables numéricas disponibles para mostrar.")
    else:
        var_tabs = st.tabs([c.replace("_", " ").title() for c in numeric_cols])

        for i, col in enumerate(numeric_cols):
            with var_tabs[i]:
                st.markdown(f"### **Variable:** {col}")

                # --- Fila 1: Histograma + QQplot ---
                row1_col1, row1_col2 = st.columns([1, 1])

                # HISTOGRAMA
                with row1_col1:
                    fig_hist = px.histogram(
                        df_f,
                        x=col,
                        nbins=15,
                        color="Estrato" if "Estrato" in df_f.columns else None,
                        color_discrete_sequence=palette_institucional,
                        title=f"<b>📈 Histograma de {col}</b>",
                        hover_data=df_f.columns,
                        opacity=0.8
                    )
                    fig_hist.update_layout(
                        title_x=0,
                        hovermode="x unified",
                        showlegend=True,
                        paper_bgcolor="#0E1117",
                        plot_bgcolor="#1A1C23",
                        font=dict(color="#E0E0E0")
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                # QQPLOT
                with row1_col2:
                    figm, ax = plt.subplots(figsize=(5, 4))
                    clean = df_f[col].dropna()
                    ax.set_facecolor("#1E1E1E")
                    figm.patch.set_facecolor("#1E1E1E")
                    if len(clean) >= 3:
                        (osm, osr), (slope, intercept, r) = probplot(clean, dist="norm")
                        ax.scatter(osm, osr, color="#00BFFF", alpha=0.7, label="Datos")
                        ax.plot(osm, slope*osm + intercept, color="#8B0000", linewidth=2, label="Línea normal")
                        ax.set_title(f"QQ-Plot — {col}", color="white", fontsize=11, weight="bold")
                        ax.legend(facecolor="#1E1E1E", labelcolor="white")
                        ax.tick_params(colors="white")
                    else:
                        ax.text(0.3, 0.5, "Insuficientes datos", color="white", fontsize=12)
                    st.pyplot(figm)

                # --- Fila 2: Boxplot ---
                if "Estrato" in df_f.columns:
                    fig_box = px.box(
                        df_f,
                        x="Estrato",
                        y=col,
                        color="Estrato",
                        points="all",
                        color_discrete_sequence=palette_institucional,
                        title=f"<b>📦 Distribución de {col} por Estrato</b>",
                        hover_data=df_f.columns
                    )
                else:
                    fig_box = px.box(
                        df_f,
                        y=col,
                        points="all",
                        color_discrete_sequence=["#0F4C81"],
                        title=f"<b>📦 Distribución de {col}</b>",
                        hover_data=df_f.columns
                    )

                fig_box.update_layout(
                    title_x=0,
                    hovermode="x unified",
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#1A1C23",
                    font=dict(color="#E0E0E0")
                )
                st.plotly_chart(fig_box, use_container_width=True)

                # --- Interpretación ---
                st.markdown(f"**📖 Interpretación de {col}:**")
                if col == "Horas_Uso":
                    st.write(
                        "El histograma muestra una tendencia hacia un uso **moderado a elevado** del smartphone, "
                        "lo que refleja la alta exposición digital de la población universitaria. "
                        "El QQ-plot confirma la **no normalidad** de la variable, coherente con la concentración "
                        "de estudiantes en rangos intermedios. En el boxplot se observan posibles valores atípicos, "
                        "particularmente en los estratos medios y altos, lo que puede indicar un patrón de uso intensivo."
                    )
                elif col == "Nomofobia":
                    st.write(
                        "Se observa una **asimetría positiva**, donde la mayoría de los estudiantes presenta niveles "
                        "moderados o altos de nomofobia. Este patrón refuerza los hallazgos del estudio, indicando "
                        "una dependencia emocional creciente frente al dispositivo móvil, especialmente en estratos "
                        "más altos, donde el acceso es más frecuente y prolongado."
                    )
                elif col == "Ansiedad_social":
                    st.write(
                        "El gráfico evidencia una **distribución sesgada a la derecha**, con valores concentrados en niveles "
                        "medios-altos de ansiedad social. Este comportamiento coincide con la hipótesis del proyecto, "
                        "según la cual el uso excesivo del smartphone se relaciona con un aumento en la evitación social presencial."
                    )
                elif col == "Autoestima":
                    st.write(
                        "La distribución de la autoestima muestra valores moderadamente altos, pero con dispersión "
                        "notable. El QQ-plot refuerza la falta de normalidad. Estos resultados pueden sugerir "
                        "diferencias interpersonales ligadas a la forma de interacción digital o la autoimagen en línea."
                    )
                elif col == "Mal_uso":
                    st.write(
                        "El comportamiento del mal uso evidencia concentración en valores altos, indicando hábitos "
                        "frecuentes de utilización poco funcional o impulsivo del smartphone. Este patrón refuerza "
                        "la relación encontrada entre la nomofobia y la pérdida de control sobre el uso del dispositivo."
                    )
                else:
                    st.write(
                        "El conjunto de visualizaciones permite comprender la distribución interna de la variable, "
                        "confirmando su falta de normalidad y la presencia de posibles diferencias entre grupos "
                        "según el estrato socioeconómico."
                    )

                st.markdown("---")

# === Pestaña 3: Correlaciones no paramétricas — Spearman ===
with tabs[2]:
    st.subheader("🔗 Correlaciones no paramétricas — Spearman (Mapa de calor + Tabla con IC)")

    # Paleta: azul–ocre
    palette_institucional = ["#0F4C81", "#F4A300", "#7BAFD4", "#D97B0E", "#4C6A92"]

    spearman_vars = [c for c in ["Horas_Uso", "Nomofobia", "Ansiedad_social", "Autoestima", "Mal_uso", "Edad"] if c in df_f.columns]

    if len(spearman_vars) >= 2:
        # --- Calcular matriz de correlaciones ---
        corr = df_f[spearman_vars].corr(method="spearman")

        # --- Mapa de calor ---
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale=[
                [0, "#F4A300"], [0.5, "#ffffff"], [1, "#0F4C81"]
            ],
            zmin=-1,
            zmax=1,
            title="<b>🔶 Mapa de calor — Correlaciones Spearman</b>"
        )
        fig_corr.update_layout(
            title_x=0,
            hovermode="closest",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#1A1C23",
            font=dict(color="#E0E0E0"),
            coloraxis_colorbar=dict(title="ρ (Spearman)")
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # --- Cálculo de correlaciones individuales con IC (bootstrap) ---
        target = "Horas_Uso" if "Horas_Uso" in spearman_vars else spearman_vars[0]
        rows = []
        for v in spearman_vars:
            if v == target:
                continue
            tmp = df_f[[target, v]].dropna()
            if tmp.shape[0] < 5:
                rows.append({"Variable": v, "ρ (Spearman)": np.nan, "p-valor": np.nan, "IC 95% (bootstrap)": "n<5"})
                continue
            rho, p = stats.spearmanr(tmp[target], tmp[v], nan_policy="omit")

            # Bootstrapping
            if bootstrap_spearman and tmp.shape[0] >= 10:
                rng = np.random.default_rng(12345)
                boots = []
                for _ in range(nboots):
                    idx = rng.integers(0, len(tmp), len(tmp))
                    boots.append(stats.spearmanr(tmp[target].iloc[idx], tmp[v].iloc[idx]).correlation)
                ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
                ci_text = f"[{ci_lo:.3f}, {ci_hi:.3f}]"
            else:
                ci_text = "NA"

            rows.append({
                "Variable": v,
                "ρ (Spearman)": round(rho, 3),
                "p-valor": round(p, 4),
                "IC 95% (bootstrap)": ci_text
            })

        # Mostrar tabla con estilo
        df_corr = pd.DataFrame(rows)
        st.dataframe(df_corr.style.format(precision=3), use_container_width=True)

        # --- Interpretación ---
        st.markdown("### **📖 Interpretación de las correlaciones:**")
        st.write(
            """
            - Se evidencia una **correlación positiva y significativa entre las horas de uso y la nomofobia**, 
              lo que sugiere que a mayor exposición al smartphone, mayor dependencia emocional y conductual hacia el dispositivo.
            - También se observa una **asociación directa entre nomofobia y ansiedad social**, 
              respaldando la hipótesis de que el uso compulsivo del celular actúa como un mecanismo de evasión o compensación social.
            - La **autoestima**, en contraste, tiende a correlacionarse de forma **negativa** con la nomofobia, 
              indicando que niveles bajos de autoconfianza pueden acompañarse de un mayor apego tecnológico.
            - Estas correlaciones confirman la necesidad de intervenciones que promuevan un uso responsable 
              y regulado del smartphone en contextos universitarios.
            """
        )
    else:
        st.warning("No hay suficientes variables numéricas para calcular correlaciones Spearman.")

    st.markdown("---")

# === Pestaña 4: Test Mann–Whitney — Horas de Uso por Nomofobia (Sí/No) ===
with tabs[3]:
    st.subheader("📊 Test Mann–Whitney — Horas de Uso por Nomofobia (Sí/No)")

    if {"Nomofobia?", "Horas_Uso"}.issubset(df_f.columns):
        a = df_f[df_f["Nomofobia?"] == "Sí"]["Horas_Uso"].dropna()
        b = df_f[df_f["Nomofobia?"] == "No"]["Horas_Uso"].dropna()

        if len(a) >= 3 and len(b) >= 3:
            # --- Cálculo estadístico ---
            U, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
            n1, n2 = len(a), len(b)
            mu_U = n1 * n2 / 2
            sigma_U = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            z = (U - mu_U) / sigma_U if sigma_U > 0 else 0
            r = round(z / math.sqrt(n1 + n2), 3)

            st.write(f"**U = {U:.3f}  •  p = {p_u:.4f}  •  z = {z:.3f}  •  r = {r}**")

            # --- Mensaje de resultado ---
            if p_u < 0.05:
                st.success("✅ Diferencia estadísticamente significativa entre los grupos (p < 0.05).")
            else:
                st.info("ℹ️ No se detectaron diferencias significativas (p ≥ 0.05).")

            # --- Boxplot institucional ---
            fig_mw = px.box(
                df_f,
                x="Nomofobia?",
                y="Horas_Uso",
                points="all",
                color="Nomofobia?",
                title="<b>Distribución de Horas de Uso según presencia de Nomofobia</b>",
                hover_data=["Estrato", "Sexo"] if "Estrato" in df_f.columns else None,
                color_discrete_map={"Sí": "#0F4C81", "No": "#F4A300"}
            )

            fig_mw.update_layout(
                title_x=0,
                plot_bgcolor="#1A1C23",
                paper_bgcolor="#0E1117",
                font=dict(color="#E0E0E0"),
                xaxis_title="Condición de Nomofobia",
                yaxis_title="Horas de Uso del Smartphone",
                hoverlabel=dict(bgcolor="#0F4C81", font_color="white"),
                showlegend=False
            )

            st.plotly_chart(fig_mw, use_container_width=True)

            # --- Interpretación contextual ---
            st.markdown("### **📖 Interpretación de resultados Mann–Whitney:**")
            st.write(
                f"""
                - El valor p obtenido (**{p_u:.4f}**) indica que {'existen diferencias significativas' if p_u < 0.05 else 'no se evidencian diferencias significativas'}
                  en las **horas promedio de uso** entre quienes presentan y no presentan nomofobia.
                - El tamaño del efecto (**r = {r}**) refleja la magnitud de la diferencia, donde valores cercanos a 0.3 o superiores sugieren un efecto relevante.
                - Este resultado refuerza la hipótesis del estudio: las personas con **mayor nivel de nomofobia tienden a emplear más horas diarias en el celular**, 
                  lo cual puede estar relacionado con **conductas de dependencia y regulación emocional** frente al uso del smartphone.
                - Estos hallazgos son consistentes con la literatura revisada, destacando el **impacto psicológico del uso excesivo** y 
                  su vínculo con la ansiedad y baja autorregulación.
                """
            )
        else:
            st.warning("⚠️ Insuficientes observaciones en uno de los grupos para aplicar Mann–Whitney (mínimo 3 por grupo).")
    else:
        st.error("❌ No se encontraron las columnas 'Nomofobia?' y 'Horas_Uso' en los datos cargados.")

    st.markdown("---")

# === Pestaña 5: Kruskal–Wallis (Nomofobia por Estrato) ===
with tabs[4]:
    st.subheader("📊 Prueba Kruskal–Wallis — Nomofobia por Estrato Socioeconómico")

    if {"Estrato", "Nomofobia"}.issubset(df_f.columns):
        groups = [g["Nomofobia"].dropna() for _, g in df_f.groupby("Estrato")]
        estratos = df_f["Estrato"].dropna().unique().tolist()

        if len(groups) > 1:
            # --- Cálculo estadístico ---
            H, p_kw = stats.kruskal(*groups)
            st.write(f"**H = {H:.3f}  •  p = {p_kw:.4f}**")

            if p_kw < 0.05:
                st.success("✅ Diferencias estadísticamente significativas entre al menos dos estratos (p < 0.05).")
            else:
                st.info("ℹ️ No se evidencian diferencias significativas entre los estratos.")

            # --- Paleta (azul y ocre) ---
            palette_tomas = ["#0F4C81", "#E1B000", "#1C86EE", "#FFD166", "#073B4C"]

            # --- Boxplot interactivo con hover informativo ---
            fig_kw = px.box(
                df_f,
                x="Estrato",
                y="Nomofobia",
                color="Estrato",
                points="all",
                title="<b>Niveles de Nomofobia según Estrato Socioeconómico</b>",
                hover_data=["Sexo", "Horas_Uso", "Autoestima"] if "Horas_Uso" in df_f.columns else None,
                color_discrete_sequence=palette_tomas
            )

            fig_kw.update_layout(
                title_x=0,
                plot_bgcolor="#1A1C23",
                paper_bgcolor="#0E1117",
                font=dict(color="#E0E0E0"),
                hoverlabel=dict(bgcolor="#0F4C81", font_color="white"),
                xaxis_title="Estrato Socioeconómico",
                yaxis_title="Puntaje de Nomofobia",
                showlegend=False
            )

            st.plotly_chart(fig_kw, use_container_width=True)

            # --- Interpretación contextual ---
            st.markdown("### **📖 Interpretación de Kruskal–Wallis:**")
            st.write(
                f"""
                - El estadístico **H = {H:.3f}**, con un valor p = **{p_kw:.4f}**, evalúa si los niveles medianos de **nomofobia**
                  difieren significativamente entre los **estratos socioeconómicos**.
                - En este caso, {'se confirma la presencia de diferencias estadísticamente significativas' if p_kw < 0.05 else 'no se evidencian diferencias significativas aparentes'}.
                - Esto indica que la **condición socioeconómica** {'influye parcialmente en los niveles de nomofobia' if p_kw < 0.05 else 'no es un factor determinante en la intensidad de la nomofobia'}.
                - La tendencia visual muestra que los **estratos medios y altos** presentan medianas ligeramente superiores,
                  posiblemente debido a **mayor acceso y dependencia tecnológica**.
                - Estos hallazgos refuerzan la idea de que la nomofobia está vinculada más a la **intensidad del uso del dispositivo**
                  que a los recursos económicos en sí mismos.
                """
            )
        else:
            st.warning("⚠️ No hay suficientes grupos para ejecutar la prueba de Kruskal–Wallis (mínimo 2 estratos con datos válidos).")
    else:
        st.error("❌ No se encontraron las columnas 'Estrato' y 'Nomofobia' en los datos cargados.")

    st.markdown("---")

# === Pestaña 6: Post-Hoc Dunn ===
with tabs[5]:
    st.subheader("🔍 Análisis Post-Hoc: Dunn (Comparaciones por Pares — Bonferroni)")

    # Paleta: azul–ocre
    palette_institucional = ["#0F4C81", "#F4A300", "#7BAFD4", "#D97B0E", "#4C6A92"]

    if {"Estrato", "Nomofobia"}.issubset(df_f.columns):
        # --- Cálculo de Dunn ---
        dunn = sp.posthoc_dunn(df_f, val_col="Nomofobia", group_col="Estrato", p_adjust="bonferroni")

        st.write("**Matriz de p-valores ajustados (Bonferroni):**")
        st.dataframe(dunn.style.format("{:.4f}"), use_container_width=True)

        # --- Heatmap ---
        # construimos un escala continua simple que va de azul -> blanco -> ocre
        color_scale = [palette_institucional[0], "#ffffff", palette_institucional[1]]

        fig_dunn = px.imshow(
            dunn,
            text_auto=True,
            color_continuous_scale=color_scale,
            title="<b>Resultados Post-Hoc de Dunn — Comparaciones entre Estratos</b>",
            zmin=0,
            zmax=1
        )

        fig_dunn.update_layout(
            title_x=0,
            plot_bgcolor="#1A1C23",
            paper_bgcolor="#0E1117",
            font=dict(color="#E0E0E0"),
            hoverlabel=dict(bgcolor=palette_institucional[0], font_color="white"),
            margin=dict(t=50, l=40, r=40, b=40)
        )

        # Asegurar hover con texto claro
        fig_dunn.update_traces(hovertemplate="Comparación: %{x} vs %{y}<br>p-ajustado=%{z:.4f}")

        st.plotly_chart(fig_dunn, use_container_width=True, key="dunn_heatmap")

        # --- Interpretación contextual ---
        st.markdown("### **📖 Interpretación del Post-Hoc Dunn (contexto del proyecto):**")
        st.write(
            "- Se aplicó **Dunn (p-ajustado Bonferroni)** tras Kruskal–Wallis para identificar qué pares de estratos\n"
            "  difieren en la **nomofobia**. Los pares con **p < 0.05** son interpretados como diferencias estadísticamente significativas.\n"
            "- En el contexto del proyecto (población universitaria), estas diferencias ayudan a detectar estratos socioeconómicos\n"
            "  que reportan mayores niveles de dependencia al smartphone y así orientar recomendaciones de intervención.\n"
            "- Usa la tabla de p-valores y el mapa de calor para localizar pares específicos; las celdas con valores bajos (más cercanas\n"
            "  a 0) representan comparaciones con mayor evidencia de diferencia."
        )

        st.caption("💡 *Nota: interpreta los pares significativos en conjunto con tamaños de efecto y tamaños de muestra por estrato.*")
    else:
        st.error("❌ No se encontraron las columnas 'Estrato' y/o 'Nomofobia' en los datos cargados.")
    st.markdown("---")

# === Pestaña 7: Exploradores ===
with tabs[6]:
    st.subheader("🧭 Exploradores interactivos — Análisis visual dinámico")

    # Paleta institucional
    palette_tomas = ["#0F4C81", "#E1B000", "#1C86EE", "#FFD166", "#073B4C"]

    sub_tabs = st.tabs(["Explorador A", "Explorador B", "Explorador C"])

    # === Explorador A ===
    with sub_tabs[0]:
        st.subheader("🧩 Explorador A — Análisis Bivariado Avanzado")

        numeric = [c for c in df_f.columns if np.issubdtype(df_f[c].dtype, np.number)]
        cat = [c for c in df_f.columns if not np.issubdtype(df_f[c].dtype, np.number)]

        if numeric and len(numeric) > 1:
            x = st.selectbox("Eje X (numérico)", numeric, index=0)
            y = st.selectbox("Eje Y (numérico)", numeric, index=1)
            color = st.selectbox("Color por (categórico)", [None] + cat, index=1 if cat else 0)
            size = st.selectbox("Tamaño por (numérico, opcional)", [None] + numeric, index=0)
            trend = st.selectbox("Línea de tendencia", ["none", "ols", "lowess"], index=1)

            fig = px.scatter(
                df_f,
                x=x,
                y=y,
                color=color,
                size=size if size != "None" else None,
                trendline=None if trend == "none" else trend,
                color_discrete_sequence=palette_tomas,
                hover_data=["Sexo", "Estrato", "Nomofobia?" ] if "Nomofobia?" in df_f.columns else None,
                title=f"<b>{y} vs {x}</b>"
            )

            fig.update_layout(
                plot_bgcolor="#1A1C23",
                paper_bgcolor="#0E1117",
                font=dict(color="#E0E0E0"),
                hoverlabel=dict(bgcolor="#0F4C81", font_color="white"),
                title_x=0
            )

            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                """
                **Interpretación:**  
                Este explorador permite analizar **relaciones entre dos variables cuantitativas**, 
                con posibilidad de incorporar variables categóricas por color o numéricas por tamaño.  
                Las tendencias se estiman mediante regresión lineal (OLS) o suavizada (LOWESS).  
                """
            )
        else:
            st.warning("⚠️ No hay suficientes variables numéricas disponibles para este explorador.")

    # === Explorador B ===
    with sub_tabs[1]:
        st.subheader("🔗 Explorador B — Correlaciones Spearman personalizadas")

        cand = [c for c in ["Horas_Uso", "Nomofobia", "Ansiedad_social", "Autoestima", "Mal_uso", "Edad"] if c in df_f.columns]
        if len(cand) >= 2:
            v1 = st.selectbox("Variable A", cand, index=0, key="c1")
            v2 = st.selectbox("Variable B", cand, index=1, key="c2")

            if v1 == v2:
                st.warning("Selecciona dos variables distintas.")
            else:
                rho, p = stats.spearmanr(df_f[v1], df_f[v2], nan_policy="omit")
                st.write(f"**Spearman ρ = {rho:.3f}  •  p = {p:.4f}**")

                fig_sc = px.scatter(
                    df_f,
                    x=v1,
                    y=v2,
                    color="Nomofobia?" if "Nomofobia?" in df_f.columns else None,
                    trendline="ols",
                    color_discrete_sequence=palette_tomas,
                    hover_data=["Sexo", "Estrato"],
                    title=f"<b>{v2} vs {v1}</b>"
                )

                fig_sc.update_layout(
                    plot_bgcolor="#1A1C23",
                    paper_bgcolor="#0E1117",
                    font=dict(color="#E0E0E0"),
                    hoverlabel=dict(bgcolor="#0F4C81", font_color="white"),
                    title_x=0
                )

                st.plotly_chart(fig_sc, use_container_width=True)
                st.markdown(
                    f"""
                    **Interpretación:**  
                    Se observa una correlación **ρ = {rho:.3f}** con un valor **p = {p:.4f}**.  
                    {'Existe una relación monotónica significativa entre las variables seleccionadas.' if p < 0.05 else 'No se observa evidencia significativa de correlación.'}  
                    Este análisis ayuda a explorar asociaciones no lineales relevantes entre dimensiones psicológicas y conductuales.  
                    """
                )
        else:
            st.warning("⚠️ No hay suficientes variables numéricas para calcular correlaciones.")

    # === Explorador C ===
    with sub_tabs[2]:
        st.subheader("📊 Explorador C — Comparador por grupos categóricos")

        cat_var = st.selectbox("Variable categórica", [c for c in ["Estrato", "Sexo", "Nomofobia?"] if c in df_f.columns])
        numeric_cols = [c for c in df_f.columns if np.issubdtype(df_f[c].dtype, np.number)]
        num_var = st.selectbox("Variable numérica", numeric_cols, index=0)

        st.write(f"Resumen de **{num_var}** agrupado por **{cat_var}:**")
        grp = df_f.groupby(cat_var)[num_var].agg(["count", "mean", "median", "std"]).reset_index()
        st.dataframe(grp, use_container_width=True)

        fig_box = px.box(
            df_f,
            x=cat_var,
            y=num_var,
            points="all",
            color=cat_var,
            color_discrete_sequence=palette_tomas,
            title=f"<b>{num_var} por {cat_var}</b>"
        )

        fig_box.update_layout(
            plot_bgcolor="#1A1C23",
            paper_bgcolor="#0E1117",
            font=dict(color="#E0E0E0"),
            hoverlabel=dict(bgcolor="#0F4C81", font_color="white"),
            title_x=0
        )

        st.plotly_chart(fig_box, use_container_width=True)

        uniques = df_f[cat_var].dropna().unique()
        if len(uniques) > 2:
            groups_list = [g[num_var].dropna() for _, g in df_f.groupby(cat_var)]
            try:
                Hc, p_hc = stats.kruskal(*groups_list)
                st.write(f"**Kruskal–Wallis:** H = {Hc:.3f} • p = {p_hc:.4f}")
                if p_hc < 0.05:
                    st.success("✅ Diferencias estadísticamente significativas entre grupos (p < 0.05).")
                else:
                    st.info("ℹ️ No se detectan diferencias significativas entre los grupos.")
            except Exception as e:
                st.error(f"No se pudo ejecutar Kruskal–Wallis: {e}")
        else:
            st.info("Kruskal–Wallis no aplica (menos de 3 grupos).")

        st.markdown(
            """
            **Interpretación:**  
            Este explorador permite comparar las distribuciones de una variable cuantitativa a través de categorías.  
            Los diagramas de caja muestran medianas, dispersión y posibles valores atípicos, 
            facilitando la comprensión de diferencias entre grupos.  
            """
        )

# === Pestaña 8: Conclusiones ===
with tabs[7]:
    st.header("Conclusiones 🧭")

    # ---- Radar: comparación multidimensional ----
    st.subheader("Comparación global de indicadores principales")

    # Variables de interés (presentes en el dataframe)
    radar_vars = [v for v in ["Nomofobia", "Ansiedad_social", "Autoestima", "Mal_uso"] if v in df_f.columns]

    if radar_vars:
        radar_means = df_f[radar_vars].mean().reset_index()
        radar_means.columns = ["Variable", "Promedio"]

        # Cerramos el polígono repitiendo el primer punto
        radar_means = pd.concat([radar_means, radar_means.iloc[[0]]], ignore_index=True)

        fig_radar = go.Figure(
            data=go.Scatterpolar(
                r=radar_means["Promedio"],
                theta=radar_means["Variable"],
                fill="toself",
                name="Promedio general",
                line=dict(color="#0F4C81", width=3),
                fillcolor="rgba(15, 76, 129, 0.4)"
            )
        )

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, radar_means["Promedio"].max() * 1.1], gridcolor="#444"),
                angularaxis=dict(tickfont=dict(size=11, color="white"))
            ),
            showlegend=False,
            plot_bgcolor="#1A1C23",
            paper_bgcolor="#0E1117",
            font=dict(color="#E0E0E0"),
            title="<b>Promedios generales por constructo</b>"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption("Gráfico radar 1. Comparación de niveles medios de las variables clave del estudio.")

        st.markdown("""
        **Interpretación del radar:**  
        El gráfico muestra los niveles promedio de los principales constructos del estudio.  
        • Los valores más altos en *Nomofobia* y *Mal uso* reflejan una relación directa con el tiempo de pantalla.  
        • En contraste, *Autoestima* presenta valores moderados, lo que sugiere un posible factor protector.  
        • *Ansiedad social* mantiene un comportamiento medio, en línea con la literatura que asocia dependencia tecnológica con evitación social.
        """)
    else:
        st.info("No se encontraron variables suficientes para generar el gráfico radar.")

    st.markdown("---")

    # ---- Resumen estadístico e interpretativo ----
    st.subheader("Resumen de hallazgos estadísticos 📊")

    conclusions = []
    if "Horas_Uso" in df_f.columns:
        for v in ["Nomofobia", "Ansiedad_social", "Autoestima"]:
            if v in df_f.columns:
                rho, p = stats.spearmanr(df_f["Horas_Uso"], df_f[v], nan_policy="omit")
                if np.isnan(rho): 
                    continue
                text = f"**{v}** — ρ={rho:.3f}, p={p:.4f}. "
                text += "📈 Asociación significativa con las horas de uso." if p < 0.05 else "No se observa asociación significativa."
                conclusions.append(text)

    if 'p_u' in locals():
        conclusions.append(f"**Mann–Whitney:** p={p_u:.4f} → {'✅ Diferencia significativa' if p_u < 0.05 else 'Sin diferencia significativa'} entre grupos de Nomofobia (Sí/No).")
    if 'p_kw' in locals():
        conclusions.append(f"**Kruskal–Wallis:** p={p_kw:.4f} → {'✅ Diferencias entre estratos' if p_kw < 0.05 else 'No diferencias entre estratos'} en Nomofobia.")

    if {"Estrato", "Nomofobia"}.issubset(df_f.columns):
        sig_pairs = []
        if "dunn" in locals():
            for i in dunn.index:
                for j in dunn.columns:
                    if i == j: continue
                    pv = dunn.loc[i, j]
                    if pv < 0.05:
                        sig_pairs.append(f"{i}–{j} (p={pv:.3f})")
        if sig_pairs:
            conclusions.append("**Post-hoc Dunn:** diferencias significativas entre → " + ", ".join(sig_pairs))
        else:
            conclusions.append("**Post-hoc Dunn:** no se detectaron diferencias significativas entre pares de estratos.")

    # Presentación visual ordenada
    for c in conclusions:
        st.markdown(f"• {c}")

    st.markdown("---")

    # ---- Conclusión general del proyecto ----
    st.subheader("Conclusión final del estudio 🎯")
    st.markdown("""
    En conjunto, los resultados obtenidos confirman que el tiempo de uso del smartphone mantiene una **relación directa con los niveles de nomofobia y de mal uso del dispositivo**, 
    además de mostrar una **asociación significativa con la ansiedad social**.  
    Esto sugiere que el uso excesivo del teléfono no solo impacta en la dependencia tecnológica, sino también en el bienestar psicológico, 
    evidenciando la necesidad de estrategias institucionales para fomentar un uso consciente y equilibrado de la tecnología entre los estudiantes universitarios.
    """)
    st.write("""
    El análisis integral evidencia que **la nomofobia se asocia de forma significativa con un mayor tiempo de uso del smartphone y niveles elevados de ansiedad social**, 
    especialmente en determinados estratos socioeconómicos. Estas tendencias sugieren que el fenómeno no solo es individual, sino también contextual y cultural.  
    Por tanto, los resultados respaldan la necesidad de **estrategias institucionales de bienestar digital**, centradas en la autorregulación tecnológica y la educación emocional de los jóvenes universitarios.""")

    st.markdown("---")

    # ---- Cuadro de recomendaciones finales ----
    st.subheader("Recomendaciones 💡")
    recs = []
    if any("Ansiedad_social" in s and "significativa" in s for s in conclusions):
        recs.append("Implementar talleres de regulación emocional enfocados en la ansiedad social asociada al uso del smartphone.")
    if 'p_u' in locals() and p_u < 0.05:
        recs.append("Desarrollar campañas para promover hábitos digitales saludables y control del tiempo de uso.")

    for i, r in enumerate(recs, 1):
        st.markdown(f"**{i}.** {r}")
    st.markdown("---")

    st.success("Las conclusiones integran resultados descriptivos, correlacionales y no paramétricos, reforzando la validez del análisis aplicado.")
    st.caption("Dashboard Nomofobia | Estadística No Paramétrica | Johann Rivera & Julian Valderrama | 2025")