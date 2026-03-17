"""
Dashboard EDA — Opiniones sobre Ciudades de Colombia
Framework QUEST: Question · Understand · Explore · Study · Tell
"""

import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN GENERAL
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDA · Ciudades de Colombia",
    page_icon="🗺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# ESTILOS PERSONALIZADOS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

/* Fondo general nude/crema */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #F5F0EA !important;
    font-family: 'DM Sans', sans-serif;
    color: #2C2825;
}

[data-testid="stSidebar"] {
    background-color: #EDE7DC !important;
    border-right: 1px solid #D9D0C5;
}

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #7A6F65 !important;
    border: none;
    padding: 0.6rem 1.2rem;
    border-radius: 0;
    background: transparent;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #2C2825 !important;
    border-bottom: 2px solid #8B6F4E !important;
    background: transparent !important;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 0.2rem;
    border-bottom: 1px solid #D9D0C5;
    background: transparent;
}

/* Métricas */
[data-testid="metric-container"] {
    background-color: #EDE7DC;
    border: 1px solid #D9D0C5;
    border-radius: 4px;
    padding: 1rem;
}
[data-testid="metric-container"] label {
    color: #7A6F65 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #2C2825 !important;
    font-family: 'Playfair Display', serif;
    font-size: 2rem !important;
}

/* Encabezados */
h1, h2, h3 {
    font-family: 'Playfair Display', serif;
    color: #2C2825;
}

/* Sidebar elementos */
[data-testid="stSidebar"] label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #7A6F65;
    font-weight: 500;
}

/* Contenedores de gráfico */
.plot-card {
    background: #FDFAF6;
    border: 1px solid #E0D8CF;
    border-radius: 6px;
    padding: 1.4rem 1.4rem 0.8rem;
    margin-bottom: 1.4rem;
}
.question-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8B6F4E;
    margin-bottom: 0.3rem;
}
.chart-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: #2C2825;
    margin-bottom: 1rem;
}
.section-intro {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.92rem;
    color: #5A5148;
    line-height: 1.75;
    max-width: 780px;
    margin-bottom: 1.8rem;
}
.divider {
    border: none;
    border-top: 1px solid #D9D0C5;
    margin: 1.5rem 0;
}
.kpi-row { margin-bottom: 1.8rem; }

/* Scrollbar sutil */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #F5F0EA; }
::-webkit-scrollbar-thumb { background: #C9BDB0; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# PALETA DE COLORES PROFESIONAL
# ──────────────────────────────────────────────────────────────
PALETTE_CITIES = [
    "#6B4F3A", "#A67C52", "#C4956A", "#D4B896", "#8B9E7A",
    "#5C7A6B", "#3D6B61", "#7A8B6F"
]
PALETTE_SEQ_WARM = ["#F5EDE0", "#E8C9A0", "#D4A574", "#B07D4A", "#8B5E3A", "#6B4226"]
PALETTE_SEQ_COOL = ["#E8EEF0", "#B8CDD4", "#7AAAB8", "#3D8099", "#1A5F73", "#0C3D4A"]
PALETTE_DIV     = ["#C1392B", "#E57C5A", "#F0C9B0", "#D9D0C5", "#7AAAB8", "#3D8099", "#1A5F73"]

COLOR_BG        = "#FDFAF6"
COLOR_GRID      = "#E8E1D8"
COLOR_TEXT      = "#2C2825"
COLOR_SUBTEXT   = "#7A6F65"
COLOR_ACCENT    = "#8B6F4E"

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font=dict(family="DM Sans, sans-serif", color=COLOR_TEXT, size=12),
        xaxis=dict(gridcolor=COLOR_GRID, linecolor=COLOR_GRID, tickfont=dict(color=COLOR_SUBTEXT)),
        yaxis=dict(gridcolor=COLOR_GRID, linecolor=COLOR_GRID, tickfont=dict(color=COLOR_SUBTEXT)),
        title=dict(font=dict(family="DM Sans", size=13, color=COLOR_TEXT)),
        colorway=PALETTE_CITIES,
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(color=COLOR_SUBTEXT, size=11)),
        margin=dict(l=50, r=30, t=40, b=50),
        hoverlabel=dict(bgcolor="#2C2825", font_color="#F5F0EA",
                        font_family="DM Sans", font_size=12),
    )
)

def apply_theme(fig, height=400):
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    fig.update_layout(height=height)
    return fig

# ──────────────────────────────────────────────────────────────
# CARGA Y PREPROCESAMIENTO DE DATOS
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(
                "opiniones_ciudades_colombia.csv",
                sep=";", encoding=enc, engine="python"
            )
            break
        except Exception:
            df = None

    if df is None:
        st.error("No se pudo cargar el archivo CSV. Asegúrate de que 'opiniones_ciudades_colombia.csv' esté en el mismo directorio.")
        st.stop()

    # Normalizar columnas
    df.columns = df.columns.str.strip()
    col_map = {c.lower(): c for c in df.columns}

    def pick(candidates):
        for cand in candidates:
            for low, real in col_map.items():
                if cand in low:
                    return real
        return None

    city_c     = pick(["ciudad", "city"])
    title_c    = pick(["título", "titulo", "title"])
    text_c     = pick(["texto", "text", "post", "contenido"])
    upvotes_c  = pick(["upvote", "score", "likes", "voto"])
    comments_c = pick(["comentario", "comment"])
    date_c     = pick(["fecha", "date", "created"])
    sub_c      = pick(["subreddit", "foro"])
    url_c      = pick(["url", "enlace"])

    df["ciudad"]      = df[city_c].astype(str).str.strip() if city_c else "Desconocida"
    df["titulo"]      = df[title_c].fillna("").astype(str) if title_c else ""
    df["texto"]       = df[text_c].fillna("").astype(str) if text_c else ""
    df["upvotes"]     = pd.to_numeric(df[upvotes_c], errors="coerce") if upvotes_c else np.nan
    df["comentarios"] = pd.to_numeric(df[comments_c], errors="coerce") if comments_c else np.nan
    df["subreddit"]   = df[sub_c].fillna("Desconocido").astype(str) if sub_c else "Desconocido"
    df["url"]         = df[url_c].fillna("").astype(str) if url_c else ""
    df["fecha"]       = pd.to_datetime(df[date_c], errors="coerce") if date_c else pd.NaT

    df["año"]  = df["fecha"].dt.year
    df["mes"]  = df["fecha"].dt.month
    df["mes_nombre"] = df["fecha"].dt.strftime("%b")
    df["periodo"] = df["fecha"].dt.to_period("M").astype(str)

    df["text_full"] = (df["titulo"] + " " + df["texto"]).str.strip()

    # Tokenización
    STOPWORDS = {
        'el','la','de','que','y','a','en','un','ser','se','no','haber','por','con','su','para',
        'es','una','como','del','muy','donde','solo','porque','eso','son','hay','desde','cuando',
        'sus','lo','ya','si','sobre','también','tambien','mas','más','pero','aunque','mientras',
        'entonces','además','ademas','luego','así','asi','sin','embargo','las','los','le','les',
        'me','te','nos','os','este','ese','esta','esa','estos','esos','estas','esas','fue',
        'todo','toda','todos','todas','cada','mucho','mucha','muchos','muchas','poco','poca',
        'uno','ella','esto','mis','alla','allá','voy','puede','siempre','entre','cual','aca',
        'etc','estan','ahora','tienen','país','pais','vida','cosas','cosa','dia','días','dias',
        'vez','hace','bien','sea','tan','era','nada','estaba','hasta','aquí','aqui','tengo',
        'estoy','soy','hola','gente','alguien','gracias','quiero','hacer','algo','años','ano',
        'qué','colombia','ciudad','ciudades','tiene','there','uno','fue','tener','the','and',
        'for','with','from','that','this','are','was','were','been','have','has','had','you',
        'your','they','them','their','our','its','into','about','than','very','just','not',
        'can','cant','could','would','al','ni','o','e','u','ah','eh','oh','eh','ya','hay'
    }

    def normalize(token):
        token = str(token).strip().lower()
        return "".join(
            ch for ch in unicodedata.normalize("NFKD", token)
            if not unicodedata.combining(ch)
        )

    def tokenize(text):
        text = re.sub(r"http\S+|www\S+", " ", str(text).lower())
        text = re.sub(r"[^a-záéíóúñü\s]", " ", text)
        return [
            normalize(t) for t in text.split()
            if len(normalize(t)) > 2 and normalize(t) not in STOPWORDS
        ]

    df["tokens"] = df["text_full"].apply(tokenize)

    # Sentimiento léxico
    POSITIVE = {
        "bueno","excelente","bonito","seguro","agradable","mejor","increible","increíble",
        "amable","progreso","oportunidad","recomendado","feliz","genial","positivo",
        "hermoso","tranquilo","limpio","organizado","desarrollado","moderno","pujante",
        "cultural","turístico","turistico","gastronomia","gastronomía"
    }
    NEGATIVE = {
        "malo","terrible","inseguro","peligroso","peor","caro","violencia","robo",
        "caos","congestion","congestión","sucio","problema","negativo","difícil","dificil",
        "trafico","tráfico","delincuencia","corrupcion","corrupción","desempleo",
        "pobreza","contaminacion","contaminación","desigual"
    }

    df["pos"] = df["tokens"].apply(lambda tks: sum(1 for t in tks if t in POSITIVE))
    df["neg"] = df["tokens"].apply(lambda tks: sum(1 for t in tks if t in NEGATIVE))
    df["sentimiento"] = np.where(df["pos"] > df["neg"], "Positivo",
                         np.where(df["neg"] > df["pos"], "Negativo", "Neutral"))

    return df

df = load_data()

# ──────────────────────────────────────────────────────────────
# SIDEBAR — FILTROS GLOBALES
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Playfair Display',serif; font-size:1.3rem;
    color:#2C2825; margin-bottom:0.2rem;">Ciudades de Colombia</div>
    <div style="font-family:'DM Sans',sans-serif; font-size:0.75rem;
    color:#8B6F4E; text-transform:uppercase; letter-spacing:0.08em;
    margin-bottom:1.5rem;">EDA · Framework QUEST</div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    ciudades_disp = sorted(df["ciudad"].unique().tolist())
    ciudades_sel = st.multiselect(
        "Ciudades", ciudades_disp, default=ciudades_disp,
        help="Selecciona las ciudades que deseas analizar"
    )

    años_disp = sorted(df["año"].dropna().unique().astype(int).tolist())
    años_sel = st.multiselect(
        "Años", años_disp, default=años_disp
    )

    meses_map = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
                 7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}
    meses_disp = sorted(df["mes"].dropna().unique().astype(int).tolist())
    meses_sel = st.multiselect(
        "Meses", [meses_map[m] for m in meses_disp],
        default=[meses_map[m] for m in meses_disp]
    )
    meses_num_sel = [k for k,v in meses_map.items() if v in meses_sel]

    subs_disp = sorted(df["subreddit"].unique().tolist())
    subs_sel = st.multiselect(
        "Subreddits", subs_disp, default=subs_disp
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:#9A8F85;line-height:1.6;'>"
        "Fuente: Reddit — publicaciones de 2020 a 2026.<br>"
        "Análisis exploratorio de datos bajo la metodología QUEST."
        "</div>",
        unsafe_allow_html=True
    )

# Aplicar filtros
mask = (
    df["ciudad"].isin(ciudades_sel) &
    df["año"].isin(años_sel) &
    df["mes"].isin(meses_num_sel) &
    df["subreddit"].isin(subs_sel)
)
dff = df[mask].copy()

# ──────────────────────────────────────────────────────────────
# CABECERA PRINCIPAL
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:0.2rem;">
  <span style="font-family:'DM Sans',sans-serif;font-size:0.75rem;
  text-transform:uppercase;letter-spacing:0.1em;color:#8B6F4E;">
  Análisis Exploratorio de Datos
  </span>
</div>
<h1 style="font-family:'Playfair Display',serif;font-size:2.2rem;
color:#2C2825;margin:0 0 0.3rem;">
  Opiniones sobre Ciudades de Colombia
</h1>
<div style="font-family:'DM Sans',sans-serif;font-size:0.88rem;
color:#7A6F65;margin-bottom:1.8rem;">
  Publicaciones de Reddit · 2020 – 2026 · 1 854 registros originales
</div>
<hr style="border:none;border-top:1px solid #D9D0C5;margin-bottom:1.8rem;">
""", unsafe_allow_html=True)

# KPIs globales
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Publicaciones", f"{len(dff):,}")
k2.metric("Ciudades", dff["ciudad"].nunique())
k3.metric("Subreddits", dff["subreddit"].nunique())
k4.metric("Upvotes totales", f"{int(dff['upvotes'].sum()):,}")
k5.metric("Comentarios totales", f"{int(dff['comentarios'].sum()):,}")

st.markdown("<hr style='border:none;border-top:1px solid #D9D0C5;margin:1.4rem 0;'>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# PESTAÑAS QUEST
# ──────────────────────────────────────────────────────────────
tab_q, tab_u, tab_e, tab_s, tab_t = st.tabs([
    "Q — Question",
    "U — Understand",
    "E — Explore",
    "S — Study",
    "T — Tell",
])

# ============================================================
# TAB Q — QUESTION
# ============================================================
with tab_q:
    st.markdown("""
    <div class="section-intro">
    Esta sección define las preguntas que guian el analisis exploratorio.
    Las preguntas fueron formuladas a partir del contexto del dataset: publicaciones
    de Reddit sobre ciudades colombianas entre 2020 y 2026. Cada pregunta tiene
    una visualizacion asociada que se desarrolla en las secciones siguientes.
    </div>
    """, unsafe_allow_html=True)

    preguntas = [
        ("P1", "Cuales son las palabras mas frecuentes en las publicaciones?",
         "Frecuencia de terminos — seccion E"),
        ("P2", "Que palabras aparecen mas en cada ciudad?",
         "Terminos por ciudad — seccion E"),
        ("P3", "Hay mas palabras con polaridad positiva o negativa?",
         "Analisis de sentimiento — seccion S"),
        ("P4", "Que ciudades generan mas debate (mas comentarios/interaccion)?",
         "Engagement por ciudad — seccion E"),
        ("P5", "En que periodos se habla mas de cada ciudad?",
         "Evolucion temporal — seccion S"),
        ("P6", "Que subreddits concentran la discusion de las ciudades principales?",
         "Distribucion por subreddit — seccion U"),
        ("P7", "Existen diferencias fuertes entre ciudades en volumen y engagement?",
         "Comparacion multivariada — seccion S"),
    ]

    for code, pregunta, ref in preguntas:
        col_a, col_b = st.columns([0.08, 0.92])
        with col_a:
            st.markdown(f"""
            <div style="font-family:'Playfair Display',serif;font-size:1.1rem;
            color:#8B6F4E;font-weight:700;margin-top:0.6rem;">{code}</div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div style="font-family:'DM Sans',sans-serif;font-size:0.94rem;
            color:#2C2825;margin-top:0.6rem;line-height:1.6;">
            {pregunta}
            <span style="color:#9A8F85;font-size:0.78rem;margin-left:0.5rem;">
            &rarr; {ref}</span></div>
            """, unsafe_allow_html=True)
        st.markdown("<hr style='border:none;border-top:1px solid #EDE7DC;margin:0.3rem 0;'>",
                    unsafe_allow_html=True)

    # Vista previa: publicaciones por ciudad como primer vistazo
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="question-label">Vista previa general</div>
    <div class="chart-title">Distribucion de publicaciones por ciudad</div>
    """, unsafe_allow_html=True)

    city_counts = dff["ciudad"].value_counts().reset_index()
    city_counts.columns = ["Ciudad", "Publicaciones"]
    fig_preview = px.bar(
        city_counts, x="Publicaciones", y="Ciudad",
        orientation="h",
        color="Publicaciones",
        color_continuous_scale=PALETTE_SEQ_WARM[::-1],
        text="Publicaciones"
    )
    fig_preview.update_traces(textposition="outside", textfont_size=11)
    fig_preview.update_coloraxes(showscale=False)
    fig_preview = apply_theme(fig_preview, 340)
    fig_preview.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig_preview, use_container_width=True)

# ============================================================
# TAB U — UNDERSTAND
# ============================================================
with tab_u:
    st.markdown("""
    <div class="section-intro">
    La seccion Understand valida la calidad estructural del dataset. Se revisan
    tipos de variables, valores faltantes, duplicados y estadisticos descriptivos
    para las variables numericas. Esta etapa asegura que el analisis posterior
    se realiza sobre una base de datos confiable.
    </div>
    """, unsafe_allow_html=True)

    # --- Tabla de calidad
    st.markdown("""
    <div class="question-label">Inspeccion de calidad</div>
    <div class="chart-title">Resumen de variables del dataset</div>
    """, unsafe_allow_html=True)

    cols_original = ["ciudad","titulo","texto","upvotes","comentarios","fecha","subreddit"]
    calidad_rows = []
    for col in cols_original:
        if col in dff.columns:
            calidad_rows.append({
                "Variable": col,
                "Tipo": str(dff[col].dtype),
                "Faltantes": int(dff[col].isna().sum()),
                "% Faltantes": round(dff[col].isna().mean()*100, 1),
                "Valores unicos": dff[col].nunique(),
            })
    calidad_df = pd.DataFrame(calidad_rows)
    st.dataframe(
        calidad_df.style.format({"% Faltantes": "{:.1f}%"}),
        use_container_width=True, hide_index=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Descriptivos numéricos
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        st.markdown("""
        <div class="question-label">Estadistica descriptiva — Upvotes</div>
        <div class="chart-title">Distribucion de upvotes por ciudad</div>
        """, unsafe_allow_html=True)

        fig_box_up = px.box(
            dff.dropna(subset=["upvotes"]),
            x="ciudad", y="upvotes",
            color="ciudad",
            color_discrete_sequence=PALETTE_CITIES,
            points=False,
        )
        fig_box_up = apply_theme(fig_box_up, 380)
        fig_box_up.update_layout(showlegend=False,
                                 xaxis_title="Ciudad",
                                 yaxis_title="Upvotes")
        st.plotly_chart(fig_box_up, use_container_width=True)

    with col_u2:
        st.markdown("""
        <div class="question-label">Estadistica descriptiva — Comentarios</div>
        <div class="chart-title">Distribucion de comentarios por ciudad</div>
        """, unsafe_allow_html=True)

        fig_box_cm = px.box(
            dff.dropna(subset=["comentarios"]),
            x="ciudad", y="comentarios",
            color="ciudad",
            color_discrete_sequence=PALETTE_CITIES,
            points=False,
        )
        fig_box_cm = apply_theme(fig_box_cm, 380)
        fig_box_cm.update_layout(showlegend=False,
                                 xaxis_title="Ciudad",
                                 yaxis_title="Comentarios")
        st.plotly_chart(fig_box_cm, use_container_width=True)

    # --- P6: Subreddits por ciudad
    st.markdown("<hr style='border:none;border-top:1px solid #D9D0C5;margin:1rem 0;'>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class="question-label">P6 — Que subreddits concentran la discusion de las ciudades principales?</div>
    <div class="chart-title">Participacion de subreddits por ciudad</div>
    """, unsafe_allow_html=True)

    top_n_cities = st.slider("Numero de ciudades a mostrar", 3, 7, 5, key="sub_slider")
    top_cities_sub = dff["ciudad"].value_counts().head(top_n_cities).index.tolist()
    sub_data = (
        dff[dff["ciudad"].isin(top_cities_sub)]
        .groupby(["ciudad", "subreddit"])
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    fig_sub = px.bar(
        sub_data, x="ciudad", y="n", color="subreddit",
        barmode="stack",
        color_discrete_sequence=PALETTE_CITIES,
        labels={"n": "Publicaciones", "ciudad": "Ciudad", "subreddit": "Subreddit"}
    )
    fig_sub = apply_theme(fig_sub, 400)
    fig_sub.update_layout(xaxis_title="Ciudad", yaxis_title="Publicaciones")
    st.plotly_chart(fig_sub, use_container_width=True)

    # --- Descriptivos tabla
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="question-label">Percentiles de variables numericas</div>
    <div class="chart-title">Resumen estadistico detallado</div>
    """, unsafe_allow_html=True)
    desc = dff[["upvotes","comentarios"]].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T
    st.dataframe(desc.style.format("{:.1f}"), use_container_width=True)

# ============================================================
# TAB E — EXPLORE
# ============================================================
with tab_e:
    st.markdown("""
    <div class="section-intro">
    La fase Explore profundiza en las variables categoricas y numericas de forma
    univariada y bivariada. Se analizan distribucion de publicaciones, engagement
    por ciudad, frecuencia de palabras y los terminos mas representativos por ciudad.
    </div>
    """, unsafe_allow_html=True)

    # P1 — Palabras mas frecuentes
    st.markdown("""
    <div class="question-label">P1 — Cuales son las palabras mas frecuentes en las publicaciones?</div>
    <div class="chart-title">Top palabras en el corpus completo</div>
    """, unsafe_allow_html=True)

    col_e_ctrl1, col_e_ctrl2 = st.columns([2,1])
    with col_e_ctrl1:
        n_words = st.slider("Numero de palabras a mostrar", 10, 40, 20, key="nw")
    with col_e_ctrl2:
        ciudad_palabras = st.selectbox(
            "Filtrar por ciudad (palabras)",
            ["Todas"] + sorted(dff["ciudad"].unique().tolist()),
            key="ciudad_palabras"
        )

    if ciudad_palabras == "Todas":
        tokens_src = dff
    else:
        tokens_src = dff[dff["ciudad"] == ciudad_palabras]

    all_words = [w for toks in tokens_src["tokens"] for w in toks]
    wf = pd.DataFrame(Counter(all_words).most_common(n_words), columns=["Palabra","Frecuencia"])

    fig_words = px.bar(
        wf, x="Frecuencia", y="Palabra", orientation="h",
        color="Frecuencia",
        color_continuous_scale=["#E8D5BF","#C4956A","#8B6F4E","#6B4226"],
        text="Frecuencia"
    )
    fig_words.update_traces(textposition="outside", textfont_size=10)
    fig_words.update_coloraxes(showscale=False)
    fig_words = apply_theme(fig_words, 520)
    fig_words.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig_words, use_container_width=True)

    st.markdown("<hr style='border:none;border-top:1px solid #D9D0C5;margin:1rem 0;'>",
                unsafe_allow_html=True)

    # P2 — Palabras por ciudad
    st.markdown("""
    <div class="question-label">P2 — Que palabras aparecen mas en cada ciudad?</div>
    <div class="chart-title">Top terminos por ciudad seleccionada</div>
    """, unsafe_allow_html=True)

    col_e2a, col_e2b = st.columns([2,1])
    with col_e2a:
        ciudad_sel_e2 = st.selectbox(
            "Ciudad", sorted(dff["ciudad"].unique().tolist()), key="ciudad_e2"
        )
    with col_e2b:
        top_n_e2 = st.slider("Top palabras", 5, 20, 12, key="topn_e2")

    wf_city = pd.DataFrame(
        Counter([w for toks in dff[dff["ciudad"]==ciudad_sel_e2]["tokens"] for w in toks]).most_common(top_n_e2),
        columns=["Palabra","Frecuencia"]
    )
    fig_wc = px.bar(
        wf_city, x="Frecuencia", y="Palabra", orientation="h",
        color="Frecuencia",
        color_continuous_scale=PALETTE_SEQ_COOL,
        text="Frecuencia"
    )
    fig_wc.update_traces(textposition="outside", textfont_size=10)
    fig_wc.update_coloraxes(showscale=False)
    fig_wc = apply_theme(fig_wc, 420)
    fig_wc.update_layout(yaxis=dict(categoryorder="total ascending"),
                         title=f"Palabras mas frecuentes — {ciudad_sel_e2}")
    st.plotly_chart(fig_wc, use_container_width=True)

    st.markdown("<hr style='border:none;border-top:1px solid #D9D0C5;margin:1rem 0;'>",
                unsafe_allow_html=True)

    # P4 — Engagement por ciudad
    st.markdown("""
    <div class="question-label">P4 — Que ciudades generan mas debate?</div>
    <div class="chart-title">Comentarios totales y promedio por ciudad</div>
    """, unsafe_allow_html=True)

    debate = (
        dff.groupby("ciudad", dropna=False)
        .agg(
            Publicaciones=("ciudad","size"),
            Comentarios_totales=("comentarios","sum"),
            Comentarios_promedio=("comentarios","mean"),
            Upvotes_totales=("upvotes","sum"),
        )
        .reset_index()
        .sort_values("Comentarios_totales", ascending=False)
    )

    metrica_debate = st.radio(
        "Metrica de engagement",
        ["Comentarios totales","Comentarios promedio","Upvotes totales"],
        horizontal=True, key="metrica_debate"
    )
    col_debate = {
        "Comentarios totales": "Comentarios_totales",
        "Comentarios promedio": "Comentarios_promedio",
        "Upvotes totales": "Upvotes_totales",
    }[metrica_debate]

    fig_debate = px.bar(
        debate.sort_values(col_debate, ascending=True),
        x=col_debate, y="ciudad", orientation="h",
        color=col_debate,
        color_continuous_scale=PALETTE_SEQ_WARM[::-1],
        text=col_debate,
        labels={col_debate: metrica_debate, "ciudad":"Ciudad"}
    )
    fig_debate.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_debate.update_coloraxes(showscale=False)
    fig_debate = apply_theme(fig_debate, 360)
    st.plotly_chart(fig_debate, use_container_width=True)

    # Tabla resumen
    with st.expander("Ver tabla completa de engagement"):
        st.dataframe(
            debate.style.format({
                "Comentarios_totales": "{:,.0f}",
                "Comentarios_promedio": "{:.1f}",
                "Upvotes_totales": "{:,.0f}",
            }),
            use_container_width=True, hide_index=True
        )

# ============================================================
# TAB S — STUDY
# ============================================================
with tab_s:
    st.markdown("""
    <div class="section-intro">
    La fase Study examina relaciones entre variables: polaridad de sentimiento,
    correlacion entre upvotes y comentarios, comportamiento temporal y comparacion
    multivariada entre ciudades.
    </div>
    """, unsafe_allow_html=True)

    # P3 — Sentimiento
    st.markdown("""
    <div class="question-label">P3 — Hay mas palabras con polaridad positiva o negativa?</div>
    <div class="chart-title">Analisis de sentimiento lexico por ciudad</div>
    """, unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        # Pie global
        sent_global = dff["sentimiento"].value_counts().reset_index()
        sent_global.columns = ["Sentimiento","Count"]
        COLOR_SENT = {"Positivo":"#5C7A6B","Negativo":"#B05B5B","Neutral":"#C4A882"}
        fig_pie = px.pie(
            sent_global, names="Sentimiento", values="Count",
            color="Sentimiento",
            color_discrete_map=COLOR_SENT,
            hole=0.45
        )
        fig_pie = apply_theme(fig_pie, 340)
        fig_pie.update_traces(textposition="outside", textinfo="percent+label")
        fig_pie.update_layout(title="Distribucion global de sentimiento",
                              showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_s2:
        # Barras apiladas por ciudad
        sent_city = (
            dff.groupby(["ciudad","sentimiento"])
            .size()
            .reset_index(name="n")
        )
        fig_sent = px.bar(
            sent_city, x="ciudad", y="n", color="sentimiento",
            barmode="stack",
            color_discrete_map=COLOR_SENT,
            labels={"n":"Publicaciones","ciudad":"Ciudad","sentimiento":"Sentimiento"}
        )
        fig_sent = apply_theme(fig_sent, 340)
        fig_sent.update_layout(title="Sentimiento por ciudad",
                               xaxis_title="", yaxis_title="Publicaciones")
        st.plotly_chart(fig_sent, use_container_width=True)

    # Balance de sentimiento
    balance = (
        dff.groupby("ciudad")[["pos","neg"]]
        .sum()
        .assign(balance=lambda x: x["pos"] - x["neg"])
        .reset_index()
        .sort_values("balance", ascending=False)
    )
    fig_balance = px.bar(
        balance, x="ciudad", y="balance",
        color="balance",
        color_continuous_scale=PALETTE_DIV,
        labels={"balance":"Balance (positivo - negativo)","ciudad":"Ciudad"},
        text="balance"
    )
    fig_balance.update_traces(texttemplate="%{text:+.0f}", textposition="outside")
    fig_balance = apply_theme(fig_balance, 320)
    fig_balance.update_layout(
        title="Balance de polaridad por ciudad",
        coloraxis_colorbar=dict(title="Balance"),
    )
    fig_balance.add_hline(y=0, line_dash="dash", line_color=COLOR_SUBTEXT, line_width=1)
    st.plotly_chart(fig_balance, use_container_width=True)

    st.markdown("<hr style='border:none;border-top:1px solid #D9D0C5;margin:1rem 0;'>",
                unsafe_allow_html=True)

    # P5 — Evolucion temporal
    st.markdown("""
    <div class="question-label">P5 — En que periodos se habla mas de cada ciudad?</div>
    <div class="chart-title">Evolucion mensual de publicaciones</div>
    """, unsafe_allow_html=True)

    col_t1, col_t2 = st.columns([2,1])
    with col_t1:
        cities_temporal = st.multiselect(
            "Ciudades para analisis temporal",
            sorted(dff["ciudad"].unique().tolist()),
            default=sorted(dff["ciudad"].unique().tolist())[:5],
            key="cities_temporal"
        )
    with col_t2:
        granularidad = st.radio("Granularidad", ["Mensual","Anual"], key="gran")

    dff_time = dff.dropna(subset=["fecha"]).copy()
    if cities_temporal:
        dff_time = dff_time[dff_time["ciudad"].isin(cities_temporal)]

    if granularidad == "Mensual":
        dff_time["periodo_plot"] = dff_time["fecha"].dt.to_period("M").astype(str)
    else:
        dff_time["periodo_plot"] = dff_time["año"].astype(str)

    serie = (
        dff_time.groupby(["periodo_plot","ciudad"])
        .size()
        .reset_index(name="Publicaciones")
    )

    fig_time = px.line(
        serie, x="periodo_plot", y="Publicaciones", color="ciudad",
        color_discrete_sequence=PALETTE_CITIES,
        markers=True,
        labels={"periodo_plot":"Periodo","ciudad":"Ciudad"}
    )
    fig_time = apply_theme(fig_time, 420)
    fig_time.update_layout(
        xaxis=dict(tickangle=-35),
        xaxis_title="Periodo", yaxis_title="Publicaciones"
    )
    st.plotly_chart(fig_time, use_container_width=True)

    st.markdown("<hr style='border:none;border-top:1px solid #D9D0C5;margin:1rem 0;'>",
                unsafe_allow_html=True)

    # P7 — Correlacion upvotes vs comentarios
    st.markdown("""
    <div class="question-label">P7 — Existen diferencias fuertes en engagement entre ciudades?</div>
    <div class="chart-title">Relacion entre upvotes y comentarios por publicacion</div>
    """, unsafe_allow_html=True)

    max_upvotes = st.slider(
        "Limitar upvotes (para mejor visualizacion)", 50, 2600, 500, step=50
    )

    scatter_df = dff.dropna(subset=["upvotes","comentarios"]).copy()
    scatter_df = scatter_df[scatter_df["upvotes"] <= max_upvotes]

    fig_scatter = px.scatter(
        scatter_df, x="upvotes", y="comentarios",
        color="ciudad",
        color_discrete_sequence=PALETTE_CITIES,
        opacity=0.65,

        hover_data={"titulo": True, "subreddit": True},
        labels={"upvotes":"Upvotes","comentarios":"Comentarios","ciudad":"Ciudad"}
    )
    fig_scatter = apply_theme(fig_scatter, 460)
    fig_scatter.update_traces(marker_size=6)
    fig_scatter.update_layout(xaxis_title="Upvotes", yaxis_title="Comentarios")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Heatmap de correlacion por ciudad
    st.markdown("""
    <div class="question-label">Correlacion entre metricas numericas</div>
    <div class="chart-title">Matriz de correlacion — upvotes y comentarios</div>
    """, unsafe_allow_html=True)

    corr_data = (
        dff.groupby("ciudad")[["upvotes","comentarios"]]
        .mean()
        .reset_index()
    )
    fig_hm = px.imshow(
        corr_data.set_index("ciudad")[["upvotes","comentarios"]].T,
        color_continuous_scale=PALETTE_SEQ_WARM,
        aspect="auto",
        text_auto=".1f",
        labels=dict(color="Promedio")
    )
    fig_hm = apply_theme(fig_hm, 200)
    fig_hm.update_layout(coloraxis_colorbar=dict(title="Promedio"))
    st.plotly_chart(fig_hm, use_container_width=True)

# ============================================================
# TAB T — TELL
# ============================================================
with tab_t:
    st.markdown("""
    <div class="section-intro">
    La fase Tell sintetiza los hallazgos del analisis exploratorio y responde
    directamente las preguntas planteadas en la fase Question. Se presentan
    los resultados mas relevantes de forma clara y accionable.
    </div>
    """, unsafe_allow_html=True)

    # KPIs de resumen
    total_pos = int(dff["pos"].sum())
    total_neg = int(dff["neg"].sum())
    ciudad_mas_pubs = dff["ciudad"].value_counts().idxmax() if len(dff) > 0 else "N/A"
    ciudad_mas_debate = (
        dff.groupby("ciudad")["comentarios"].sum().idxmax()
        if dff["comentarios"].notna().any() else "N/A"
    )
    top_subreddit = dff["subreddit"].value_counts().idxmax() if len(dff) > 0 else "N/A"

    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    col_t1.metric("Palabras positivas", f"{total_pos:,}")
    col_t2.metric("Palabras negativas", f"{total_neg:,}")
    col_t3.metric("Ciudad mas activa", ciudad_mas_pubs)
    col_t4.metric("Ciudad mas debatida", ciudad_mas_debate)

    st.markdown("<br>", unsafe_allow_html=True)

    # Grafico consolidado: ranking final ciudades
    st.markdown("""
    <div class="question-label">Resumen consolidado</div>
    <div class="chart-title">Ranking de ciudades — publicaciones, comentarios y upvotes</div>
    """, unsafe_allow_html=True)

    ranking = (
        dff.groupby("ciudad")
        .agg(
            Publicaciones=("ciudad","size"),
            Comentarios=("comentarios","sum"),
            Upvotes=("upvotes","sum")
        )
        .reset_index()
        .sort_values("Publicaciones", ascending=False)
    )

    fig_ranking = go.Figure()
    metricas_rank = ["Publicaciones","Comentarios","Upvotes"]
    colores_rank  = [PALETTE_CITIES[0], PALETTE_CITIES[2], PALETTE_CITIES[4]]

    for met, col in zip(metricas_rank, colores_rank):
        fig_ranking.add_trace(go.Bar(
            name=met,
            x=ranking["ciudad"],
            y=ranking[met],
            marker_color=col,
            opacity=0.88,
        ))

    fig_ranking.update_layout(barmode="group")
    fig_ranking = apply_theme(fig_ranking, 420)
    fig_ranking.update_layout(xaxis_title="", yaxis_title="Cantidad",
                              legend=dict(orientation="h", yanchor="bottom",
                                          y=1.01, xanchor="right", x=1))
    st.plotly_chart(fig_ranking, use_container_width=True)

    st.markdown("<hr style='border:none;border-top:1px solid #D9D0C5;margin:1rem 0;'>",
                unsafe_allow_html=True)

    # Respuestas narrativas a cada pregunta
    st.markdown("""
    <div class="question-label">Respuestas a las preguntas QUEST</div>
    <div class="chart-title">Conclusiones del analisis exploratorio</div>
    """, unsafe_allow_html=True)

    # Calculos para las respuestas
    top5_words = [w for w,_ in Counter(
        [word for toks in dff["tokens"] for word in toks]
    ).most_common(5)]

    sent_balance_txt = (
        "predominan las expresiones positivas" if total_pos > total_neg
        else "predominan las expresiones negativas" if total_neg > total_pos
        else "hay un equilibrio entre positivo y negativo"
    )

    if not dff.dropna(subset=["fecha"]).empty:
        periodo_pico_global = (
            dff.dropna(subset=["fecha"])
            .groupby("periodo")
            .size()
            .idxmax()
        )
    else:
        periodo_pico_global = "No disponible"

    respuestas = [
        ("P1", "Cuales son las palabras mas frecuentes?",
         f"Los terminos con mayor presencia en el corpus son: {', '.join(top5_words)}. "
         f"Estos reflejan los temas centrales de conversacion en Reddit sobre las ciudades colombianas."),
        ("P2", "Que palabras destacan en cada ciudad?",
         "Cada ciudad presenta un vocabulario particular segun sus caracteristicas. "
         "El analisis por ciudad en la seccion Explore permite identificar terminos diferenciadores "
         "que revelan los temas de interes locales."),
        ("P3", "Hay mas polaridad positiva o negativa?",
         f"En el corpus filtrado, {sent_balance_txt} (positivas: {total_pos:,} — negativas: {total_neg:,}). "
         "La distribucion varia segun la ciudad y el periodo, reflejando diferentes percepciones urbanas."),
        ("P4", "Que ciudades generan mas debate?",
         f"La ciudad con mayor volumen de comentarios es {ciudad_mas_debate}, "
         "lo que indica que sus publicaciones generan mayor interaccion y discusion en la plataforma."),
        ("P5", "En que periodos se habla mas de cada ciudad?",
         f"El periodo con mayor actividad global en los datos filtrados es {periodo_pico_global}. "
         "La evolucion temporal en la seccion Study permite identificar picos por ciudad."),
        ("P6", "Que subreddits concentran la discusion?",
         f"El subreddit con mayor participacion es r/{top_subreddit}. "
         "La distribucion por foro varia segun la ciudad: algunas cuentan con subreddits dedicados "
         "mientras otras se discuten principalmente en r/Colombia."),
        ("P7", "Existen diferencias fuertes entre ciudades?",
         f"Si. Bogota y Medellin concentran el mayor volumen de publicaciones y engagement, "
         "mientras ciudades como Pereira y Bucaramanga tienen presencia significativamente menor. "
         "La correlacion entre upvotes y comentarios es positiva pero moderada."),
    ]

    for code, pregunta, respuesta in respuestas:
        col_ra, col_rb = st.columns([0.08, 0.92])
        with col_ra:
            st.markdown(f"""
            <div style="font-family:'Playfair Display',serif;font-size:1rem;
            color:#8B6F4E;font-weight:700;margin-top:0.8rem;">{code}</div>
            """, unsafe_allow_html=True)
        with col_rb:
            st.markdown(f"""
            <div style="margin-top:0.8rem;">
              <div style="font-family:'DM Sans',sans-serif;font-size:0.82rem;
              text-transform:uppercase;letter-spacing:0.06em;color:#7A6F65;
              margin-bottom:0.2rem;">{pregunta}</div>
              <div style="font-family:'DM Sans',sans-serif;font-size:0.93rem;
              color:#2C2825;line-height:1.7;">{respuesta}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<hr style='border:none;border-top:1px solid #EDE7DC;margin:0.4rem 0;'>",
                    unsafe_allow_html=True)

    # Tabla de periodos pico
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="question-label">Periodos pico por ciudad</div>
    <div class="chart-title">Mes con mayor actividad para cada ciudad</div>
    """, unsafe_allow_html=True)

    pico_por_ciudad = (
        dff.dropna(subset=["fecha"])
        .groupby(["ciudad","periodo"])
        .size()
        .reset_index(name="publicaciones")
        .sort_values("publicaciones", ascending=False)
        .groupby("ciudad")
        .first()
        .reset_index()
        .sort_values("publicaciones", ascending=False)
    )
    st.dataframe(
        pico_por_ciudad[["ciudad","periodo","publicaciones"]]
        .rename(columns={"ciudad":"Ciudad","periodo":"Periodo pico",
                          "publicaciones":"Publicaciones en el pico"}),
        use_container_width=True, hide_index=True
    )

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'DM Sans',sans-serif;font-size:0.78rem;
    color:#9A8F85;text-align:center;padding:1rem 0;
    border-top:1px solid #D9D0C5;">
    EDA · Ciudades de Colombia · Framework QUEST · Grupo 2 · Modelos Analiticos
    </div>
    """, unsafe_allow_html=True)
