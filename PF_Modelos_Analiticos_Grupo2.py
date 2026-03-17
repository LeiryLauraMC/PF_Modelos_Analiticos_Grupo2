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
# PALETA — Tonos pastel
# ──────────────────────────────────────────────────────────────
C_TEAL_DARK   = "#6BAED6"   # azul pastel medio
C_TEAL_MID    = "#9ECAE1"   # azul pastel claro
C_TEAL_LIGHT  = "#C6DBEF"   # azul pastel muy claro
C_CREAM       = "#A1D99B"   # verde salvia pastel
C_TERRA_LIGHT = "#FDBE85"   # durazno pastel
C_TERRA_DARK  = "#E09BB5"   # rosa pastel
C_BRICK       = "#BCBDDC"   # lavanda pastel
C_EARTH_MID   = "#F4C07A"   # amarillo arena pastel
C_EARTH_LIGHT = "#B5C9A1"   # verde oliva pastel

PALETTE_CITIES = [
    C_TEAL_DARK, C_TERRA_DARK, C_TERRA_LIGHT,
    C_TEAL_MID, C_BRICK, C_EARTH_MID, C_CREAM,
]
PALETTE_SEQ_TEAL  = ["#EFF6FB", "#C6DBEF", C_TEAL_LIGHT, C_TEAL_MID, C_TEAL_DARK, "#3182BD"]
PALETTE_SEQ_TERRA = ["#FFF7EC", "#FEE8C5", C_EARTH_MID, C_TERRA_LIGHT, "#E06010", "#7F2704"]
PALETTE_DIV       = ["#D6604D", C_TERRA_LIGHT, C_EARTH_MID, "#F7F7F7", C_TEAL_LIGHT, C_TEAL_MID, C_TEAL_DARK]

COLOR_BG      = "#F5F0EA"
COLOR_CARD    = "#FDFAF6"
COLOR_GRID    = "#E8E1D8"
COLOR_TEXT    = "#2C2825"
COLOR_SUBTEXT = "#7A6F65"

# ──────────────────────────────────────────────────────────────
# ESTILOS
# ──────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
    background-color: {COLOR_BG} !important;
    font-family: 'DM Sans', sans-serif;
    color: {COLOR_TEXT};
}}
[data-testid="stSidebar"] {{
    background-color: #EDE7DC !important;
    border-right: 1px solid #D9D0C5;
}}
[data-testid="stTabs"] button {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {COLOR_SUBTEXT} !important;
    border: none;
    padding: 0.6rem 1.2rem;
    border-radius: 0;
    background: transparent;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {COLOR_TEXT} !important;
    border-bottom: 2px solid {C_TEAL_DARK} !important;
    background: transparent !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    gap: 0.2rem;
    border-bottom: 1px solid #D9D0C5;
    background: transparent;
}}
[data-testid="metric-container"] {{
    background-color: #EDE7DC;
    border: 1px solid #D9D0C5;
    border-radius: 4px;
    padding: 1rem;
}}
[data-testid="metric-container"] label {{
    color: {COLOR_SUBTEXT} !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {COLOR_TEXT} !important;
    font-family: 'Playfair Display', serif;
    font-size: 2rem !important;
}}
h1, h2, h3 {{
    font-family: 'Playfair Display', serif;
    color: {COLOR_TEXT};
}}
[data-testid="stSidebar"] label {{
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {COLOR_SUBTEXT};
    font-weight: 500;
}}
.question-label {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {C_TEAL_DARK};
    margin-bottom: 0.3rem;
}}
.chart-title {{
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: {COLOR_TEXT};
    margin-bottom: 1rem;
}}
.section-intro {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.92rem;
    color: #5A5148;
    line-height: 1.75;
    text-align: justify;
    width: 100%;
    margin-bottom: 1.8rem;
}}
.respuesta-txt {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.93rem;
    color: {COLOR_TEXT};
    line-height: 1.7;
    text-align: justify;
}}
.pregunta-txt {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {COLOR_SUBTEXT};
    margin-bottom: 0.2rem;
}}
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {COLOR_BG}; }}
::-webkit-scrollbar-thumb {{ background: #C9BDB0; border-radius: 3px; }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# TEMA PLOTLY
# ──────────────────────────────────────────────────────────────
def apply_theme(fig, height=400):
    fig.update_layout(
        paper_bgcolor=COLOR_CARD,
        plot_bgcolor=COLOR_CARD,
        font=dict(family="DM Sans, sans-serif", color=COLOR_TEXT, size=12),
        xaxis=dict(gridcolor=COLOR_GRID, linecolor=COLOR_GRID,
                   tickfont=dict(color=COLOR_SUBTEXT)),
        yaxis=dict(gridcolor=COLOR_GRID, linecolor=COLOR_GRID,
                   tickfont=dict(color=COLOR_SUBTEXT)),
        colorway=PALETTE_CITIES,
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(color=COLOR_SUBTEXT, size=11)),
        margin=dict(l=50, r=30, t=45, b=50),
        hoverlabel=dict(bgcolor="#2C2825", font_color="#F5F0EA",
                        font_family="DM Sans", font_size=12),
        height=height,
    )
    return fig





# ──────────────────────────────────────────────────────────────
# CARGA Y PREPROCESAMIENTO
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = None
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(
                "opiniones_ciudades_colombia.csv",
                sep=";", encoding=enc, engine="python",
            )
            break
        except Exception:
            pass
    if df is None:
        st.error("No se pudo cargar 'opiniones_ciudades_colombia.csv'.")
        st.stop()

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

    df["año"]     = df["fecha"].dt.year
    df["mes"]     = df["fecha"].dt.month
    df["periodo"] = df["fecha"].dt.to_period("M").astype(str)
    df["text_full"] = (df["titulo"] + " " + df["texto"]).str.strip()

    STOPWORDS = {
        'el','la','de','que','y','a','en','un','ser','se','no','haber','por','con','su','para',
        'es','una','como','del','muy','donde','solo','porque','eso','son','hay','desde','cuando',
        'sus','lo','ya','si','sobre','también','tambien','mas','más','pero','aunque','mientras',
        'entonces','además','ademas','luego','así','asi','sin','embargo','las','los','le','les',
        'me','te','nos','os','este','ese','esta','esa','estos','esos','estas','esas','fue',
        'todo','toda','todos','todas','cada','mucho','mucha','muchos','muchas','poco','poca',
        'uno','ella','esto','mis','alla','voy','puede','siempre','entre','cual','aca','etc',
        'estan','ahora','tienen','pais','vida','cosas','cosa','dia','dias','vez','hace','bien',
        'sea','tan','era','nada','estaba','hasta','aqui','tengo','estoy','soy','hola','gente',
        'alguien','gracias','quiero','hacer','algo','anos','ano','colombia','ciudad','ciudades',
        'tiene','the','and','for','with','from','that','this','are','was','were','been','have',
        'has','had','you','your','they','them','their','our','its','into','about','than','very',
        'just','not','can','cant','could','would','al','ni','o','e','u','ah','eh','oh','hay',
        'van','ver','ir','han','otro','otra','otros','otras','mismo','misma','alli','aca',
        'tampoco','nunca','despues','antes','aunque','porque','cuando','donde','como','cual',
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

    POSITIVE = {
        "bueno","excelente","bonito","seguro","agradable","mejor","increible",
        "amable","progreso","oportunidad","recomendado","feliz","genial","positivo",
        "hermoso","tranquilo","limpio","organizado","desarrollado","moderno","pujante",
        "cultural","turistico","gastronomia","maravilloso","lindo","chevere",
    }
    NEGATIVE = {
        "malo","terrible","inseguro","peligroso","peor","caro","violencia","robo",
        "caos","congestion","sucio","problema","negativo","dificil","trafico",
        "delincuencia","corrupcion","desempleo","pobreza","contaminacion","desigual",
        "crimen","peligro","miedo","inseguridad",
    }

    df["pos"] = df["tokens"].apply(lambda tks: sum(1 for t in tks if t in POSITIVE))
    df["neg"] = df["tokens"].apply(lambda tks: sum(1 for t in tks if t in NEGATIVE))
    df["sentimiento"] = np.where(df["pos"] > df["neg"], "Positivo",
                         np.where(df["neg"] > df["pos"], "Negativo", "Neutral"))
    return df


df = load_data()

# ──────────────────────────────────────────────────────────────
# ESTILOS EXTRA PARA FILTROS PERSONALIZADOS
# ──────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Checkboxes del sidebar — texto visible ── */
[data-testid="stSidebar"] [data-testid="stCheckbox"] label p {{
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    color: {COLOR_TEXT} !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    font-weight: 400 !important;
}}
[data-testid="stSidebar"] [data-testid="stCheckbox"] input[type="checkbox"] {{
    accent-color: {C_TEAL_DARK} !important;
    width: 14px;
    height: 14px;
}}
/* ── Expanders del sidebar ── */
[data-testid="stSidebar"] [data-testid="stExpander"] {{
    background-color: #E8E0D5 !important;
    border: 1px solid #D4CAC0 !important;
    border-radius: 4px !important;
    margin-bottom: 0.5rem !important;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] summary p {{
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: {COLOR_TEXT} !important;
    font-weight: 500 !important;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover p {{
    color: {C_TEAL_DARK} !important;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
    padding: 0.3rem 0.6rem 0.6rem !important;
    background: #EDE7DC !important;
}}
/* ── Botones Todos / Ninguno ── */
[data-testid="stSidebar"] [data-testid="stButton"] button {{
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: {C_TEAL_DARK} !important;
    background: transparent !important;
    border: 1px solid {C_TEAL_DARK} !important;
    border-radius: 2px !important;
    padding: 0.15rem 0.5rem !important;
    width: 100%;
    transition: all 0.15s ease;
}}
[data-testid="stSidebar"] [data-testid="stButton"] button:hover {{
    background: {C_TEAL_DARK} !important;
    color: #FDFAF6 !important;
}}
</style>
""", unsafe_allow_html=True)


# ── Helpers para filtros con checkboxes ──────────────────────
def filter_section(label, options, key_prefix):
    """
    Expander con checkboxes. Los botones Todos/Ninguno modifican
    el session_state de cada checkbox individual ANTES de renderizarlos,
    de modo que el re-run siguiente los muestre correctamente.
    """
    options = list(options)

    # Inicializar estado de cada checkbox individualmente
    for opt in options:
        ck = f"{key_prefix}_cb_{opt}"
        if ck not in st.session_state:
            st.session_state[ck] = True

    # Callbacks que operan sobre los checkboxes individuales
    def select_all():
        for opt in options:
            st.session_state[f"{key_prefix}_cb_{opt}"] = True

    def select_none():
        for opt in options:
            st.session_state[f"{key_prefix}_cb_{opt}"] = False

    with st.expander(label, expanded=True):
        c1, c2 = st.columns(2)
        c1.button("Todos",   key=f"{key_prefix}_all",  on_click=select_all)
        c2.button("Ninguno", key=f"{key_prefix}_none", on_click=select_none)

        selected = []
        for opt in options:
            val = st.checkbox(
                str(opt),
                key=f"{key_prefix}_cb_{opt}",
            )
            if val:
                selected.append(opt)

        n, t = len(selected), len(options)
        st.markdown(
            f"<div style='font-size:0.7rem;color:{COLOR_SUBTEXT};"
            f"margin-top:0.2rem;text-align:right;'>"
            f"{n} de {t} seleccionados</div>",
            unsafe_allow_html=True,
        )
    return selected


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
meses_map = {
    1:"Enero", 2:"Febrero", 3:"Marzo", 4:"Abril",
    5:"Mayo", 6:"Junio", 7:"Julio", 8:"Agosto",
    9:"Septiembre", 10:"Octubre", 11:"Noviembre", 12:"Diciembre",
}

with st.sidebar:
    # ── Encabezado ──
    st.markdown(f"""
    <div style="font-family:'Playfair Display',serif;font-size:1.3rem;
    color:{COLOR_TEXT};margin-bottom:0.2rem;">Ciudades de Colombia</div>
    <div style="font-family:'DM Sans',sans-serif;font-size:0.75rem;
    color:{C_TEAL_DARK};text-transform:uppercase;letter-spacing:0.08em;
    margin-bottom:0.9rem;">EDA · Framework QUEST</div>
    <div style="font-family:'DM Sans',sans-serif;font-size:0.82rem;
    color:{COLOR_SUBTEXT};line-height:1.8;margin-bottom:1.2rem;
    border-left:2px solid {C_TEAL_DARK};padding-left:0.7rem;">
    Ferneys Araujo<br>Leiry L. Mares<br>María A. Pérez<br>Dana V. Ramírez
    </div>
    <div style="border-top:1px solid #D4CAC0;margin-bottom:1rem;"></div>
    <div style="font-family:'DM Sans',sans-serif;font-size:0.68rem;
    text-transform:uppercase;letter-spacing:0.1em;color:{C_TEAL_DARK};
    margin-bottom:0.7rem;">Filtros</div>
    """, unsafe_allow_html=True)

    # ── Filtros ──
    ciudades_disp = sorted(df["ciudad"].unique().tolist())
    ciudades_sel = filter_section("Ciudades", ciudades_disp, "ciudad")

    años_disp = sorted(df["año"].dropna().unique().astype(int).tolist())
    años_sel = filter_section("Años", años_disp, "anio")

    meses_disp = sorted(df["mes"].dropna().unique().astype(int).tolist())
    meses_nombres_disp = [meses_map[m] for m in meses_disp]
    meses_nombres_sel = filter_section("Meses", meses_nombres_disp, "mes")
    meses_num_sel = [k for k, v in meses_map.items() if v in meses_nombres_sel]

    subs_disp = sorted(df["subreddit"].unique().tolist())
    subs_sel = filter_section("Subreddits", subs_disp, "sub")

    # ── Pie ──
    st.markdown(
        f"<div style='border-top:1px solid #D4CAC0;margin:1rem 0 0.6rem;'></div>"
        f"<div style='font-size:0.7rem;color:#9A8F85;line-height:1.6;'>"
        "Fuente: Reddit · 2020 – 2026.<br>"
        "Metodología QUEST.</div>",
        unsafe_allow_html=True,
    )

# Filtro global
mask = (
    df["ciudad"].isin(ciudades_sel) &
    df["año"].isin(años_sel) &
    df["mes"].isin(meses_num_sel) &
    df["subreddit"].isin(subs_sel)
)
dff = df[mask].copy()

# ──────────────────────────────────────────────────────────────
# CABECERA
# ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-bottom:0.2rem;">
  <span style="font-family:'DM Sans',sans-serif;font-size:0.75rem;
  text-transform:uppercase;letter-spacing:0.1em;color:{C_TEAL_DARK};">
  Análisis Exploratorio de Datos
  </span>
</div>
<h1 style="font-family:'Playfair Display',serif;font-size:2.2rem;
color:{COLOR_TEXT};margin:0 0 0.3rem;">
  Opiniones sobre Ciudades de Colombia
</h1>
<div style="font-family:'DM Sans',sans-serif;font-size:0.88rem;
color:{COLOR_SUBTEXT};margin-bottom:1.8rem;">
  Publicaciones de Reddit · 2020 – 2026 · 1 854 registros originales
</div>
<hr style="border:none;border-top:1px solid #D9D0C5;margin-bottom:1.8rem;">
""", unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Publicaciones",       f"{len(dff):,}")
k2.metric("Ciudades",            dff["ciudad"].nunique())
k3.metric("Subreddits",          dff["subreddit"].nunique())
k4.metric("Upvotes totales",     f"{int(dff['upvotes'].sum()):,}")
k5.metric("Comentarios totales", f"{int(dff['comentarios'].sum()):,}")

st.markdown("<hr style='border:none;border-top:1px solid #D9D0C5;margin:1.4rem 0;'>",
            unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# PESTAÑAS
# ──────────────────────────────────────────────────────────────
tab_q, tab_u, tab_e, tab_s, tab_t = st.tabs([
    "Q — Question", "U — Understand", "E — Explore", "S — Study", "T — Tell",
])

# ============================================================
# Q — QUESTION
# ============================================================
with tab_q:
    st.markdown("""
    <div class="section-intro">
    Esta sección define las preguntas que guían el análisis exploratorio.
    Las preguntas fueron formuladas a partir del contexto del dataset: publicaciones
    de Reddit sobre ciudades colombianas entre 2020 y 2026. Cada pregunta tiene
    una visualización asociada que se desarrolla en las secciones siguientes
    del framework QUEST.
    </div>
    """, unsafe_allow_html=True)

    preguntas = [
        ("P1", "¿Cuáles son las palabras más frecuentes en las publicaciones?",
         "Frecuencia de términos — sección E"),
        ("P2", "¿Qué palabras aparecen más en cada ciudad?",
         "Términos por ciudad — sección E"),
        ("P3", "¿Hay más palabras con polaridad positiva o negativa?",
         "Análisis de sentimiento — sección S"),
        ("P4", "¿Qué ciudades generan más debate (más comentarios e interacción)?",
         "Engagement por ciudad — sección E"),
        ("P5", "¿En qué períodos se habla más de cada ciudad?",
         "Evolución temporal — sección S"),
        ("P6", "¿Qué subreddits concentran la discusión de las ciudades principales?",
         "Distribución por subreddit — sección U"),
        ("P7", "¿Existen diferencias fuertes entre ciudades en volumen y engagement?",
         "Comparación multivariada — sección S"),
    ]

    for code, pregunta, ref in preguntas:
        col_a, col_b = st.columns([0.08, 0.92])
        with col_a:
            st.markdown(f"""
            <div style="font-family:'Playfair Display',serif;font-size:1.1rem;
            color:{C_TEAL_DARK};font-weight:700;margin-top:0.7rem;">{code}</div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div style="font-family:'DM Sans',sans-serif;font-size:0.94rem;
            color:{COLOR_TEXT};margin-top:0.7rem;line-height:1.6;text-align:justify;">
            {pregunta}
            <span style="color:#9A8F85;font-size:0.78rem;margin-left:0.5rem;">
            &rarr; {ref}</span></div>
            """, unsafe_allow_html=True)
        st.markdown(
            "<hr style='border:none;border-top:1px solid #EDE7DC;margin:0.3rem 0;'>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="question-label">Vista previa general</div>
    <div class="chart-title">Distribución de publicaciones por ciudad</div>
    """, unsafe_allow_html=True)

    city_counts = dff["ciudad"].value_counts().reset_index()
    city_counts.columns = ["Ciudad", "Publicaciones"]
    fig_preview = px.bar(
        city_counts, x="Publicaciones", y="Ciudad", orientation="h",
        color="Publicaciones", color_continuous_scale="RdBu",
        text="Publicaciones",
    )
    fig_preview.update_traces(textposition="outside", textfont_size=11)
    fig_preview.update_coloraxes(showscale=False)
    fig_preview = apply_theme(fig_preview, 340)
    fig_preview.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig_preview, use_container_width=True)

# ============================================================
# U — UNDERSTAND
# ============================================================
with tab_u:
    st.markdown("""
    <div class="section-intro">
    La sección Understand valida la calidad estructural del dataset. Se revisan
    los tipos de variables, los valores faltantes, los duplicados y los estadísticos
    descriptivos para las variables numéricas. Esta etapa garantiza que el análisis
    posterior se realice sobre una base de datos confiable y bien comprendida.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="question-label">Inspección de calidad</div>
    <div class="chart-title">Resumen de variables del dataset</div>
    """, unsafe_allow_html=True)

    cols_original = ["ciudad", "titulo", "texto", "upvotes", "comentarios", "fecha", "subreddit"]
    calidad_rows = []
    for col in cols_original:
        if col in dff.columns:
            calidad_rows.append({
                "Variable":       col,
                "Tipo":           str(dff[col].dtype),
                "Faltantes":      int(dff[col].isna().sum()),
                "% Faltantes":    round(dff[col].isna().mean() * 100, 1),
                "Valores únicos": dff[col].nunique(),
            })
    calidad_df = pd.DataFrame(calidad_rows)
    st.dataframe(
        calidad_df.style.format({"% Faltantes": "{:.1f}%"}),
        use_container_width=True, hide_index=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col_u1, col_u2 = st.columns(2)

    with col_u1:
        st.markdown(f"""
        <div class="question-label">Estadística descriptiva — Upvotes</div>
        <div class="chart-title">Distribución de upvotes por ciudad</div>
        """, unsafe_allow_html=True)
        fig_box_up = px.box(
            dff.dropna(subset=["upvotes"]), x="ciudad", y="upvotes",
            color="ciudad", color_discrete_sequence=PALETTE_CITIES, points=False,
        )
        fig_box_up = apply_theme(fig_box_up, 380)
        fig_box_up.update_layout(showlegend=False,
                                 xaxis_title="Ciudad", yaxis_title="Upvotes")
        st.plotly_chart(fig_box_up, use_container_width=True)

    with col_u2:
        st.markdown(f"""
        <div class="question-label">Estadística descriptiva — Comentarios</div>
        <div class="chart-title">Distribución de comentarios por ciudad</div>
        """, unsafe_allow_html=True)
        fig_box_cm = px.box(
            dff.dropna(subset=["comentarios"]), x="ciudad", y="comentarios",
            color="ciudad", color_discrete_sequence=PALETTE_CITIES, points=False,
        )
        fig_box_cm = apply_theme(fig_box_cm, 380)
        fig_box_cm.update_layout(showlegend=False,
                                 xaxis_title="Ciudad", yaxis_title="Comentarios")
        st.plotly_chart(fig_box_cm, use_container_width=True)

    st.markdown(
        "<hr style='border:none;border-top:1px solid #D9D0C5;margin:1rem 0;'>",
        unsafe_allow_html=True,
    )

    st.markdown(f"""
    <div class="question-label">P6 — ¿Qué subreddits concentran la discusión de las ciudades principales?</div>
    <div class="chart-title">Participación de subreddits por ciudad</div>
    """, unsafe_allow_html=True)

    top_n_cities = st.slider("Número de ciudades a mostrar", 3, 7, 5, key="sub_slider")
    top_cities_sub = dff["ciudad"].value_counts().head(top_n_cities).index.tolist()
    sub_data = (
        dff[dff["ciudad"].isin(top_cities_sub)]
        .groupby(["ciudad", "subreddit"]).size()
        .reset_index(name="n").sort_values("n", ascending=False)
    )
    fig_sub = px.bar(
        sub_data, x="ciudad", y="n", color="subreddit", barmode="stack",
        color_discrete_sequence=PALETTE_CITIES,
        labels={"n": "Publicaciones", "ciudad": "Ciudad", "subreddit": "Subreddit"},
    )
    fig_sub = apply_theme(fig_sub, 400)
    fig_sub.update_layout(xaxis_title="Ciudad", yaxis_title="Publicaciones")
    st.plotly_chart(fig_sub, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="question-label">Percentiles de variables numéricas</div>
    <div class="chart-title">Resumen estadístico detallado</div>
    """, unsafe_allow_html=True)
    desc = dff[["upvotes", "comentarios"]].describe(
        percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
    st.dataframe(desc.style.format("{:.1f}"), use_container_width=True)

# ============================================================
# E — EXPLORE
# ============================================================
with tab_e:
    st.markdown("""
    <div class="section-intro">
    La fase Explore profundiza en las variables categóricas y numéricas de forma
    univariada y bivariada. Se analizan la distribución de publicaciones, el engagement
    por ciudad, la frecuencia de palabras globales y los términos más representativos
    en cada una de las ciudades del dataset.
    </div>
    """, unsafe_allow_html=True)

    # P1 — Palabras más frecuentes
    st.markdown(f"""
    <div class="question-label">P1 — ¿Cuáles son las palabras más frecuentes en las publicaciones?</div>
    <div class="chart-title">Top palabras en el corpus completo</div>
    """, unsafe_allow_html=True)

    col_e_ctrl1, col_e_ctrl2 = st.columns([2, 1])
    with col_e_ctrl1:
        n_words = st.slider("Número de palabras a mostrar", 10, 40, 20, key="nw")
    with col_e_ctrl2:
        ciudad_palabras = st.selectbox(
            "Filtrar por ciudad (palabras)",
            ["Todas"] + sorted(dff["ciudad"].unique().tolist()),
            key="ciudad_palabras",
        )

    tokens_src = dff if ciudad_palabras == "Todas" else dff[dff["ciudad"] == ciudad_palabras]
    all_words = [w for toks in tokens_src["tokens"] for w in toks]
    wf = pd.DataFrame(Counter(all_words).most_common(n_words), columns=["Palabra", "Frecuencia"])

    fig_words = px.bar(
        wf, x="Frecuencia", y="Palabra", orientation="h",
        color="Frecuencia", color_continuous_scale="RdBu_r",
        text="Frecuencia",
    )
    fig_words.update_traces(textposition="outside", textfont_size=10)
    fig_words.update_coloraxes(showscale=False)
    fig_words = apply_theme(fig_words, 520)
    fig_words.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig_words, use_container_width=True)

    st.markdown(
        "<hr style='border:none;border-top:1px solid #D9D0C5;margin:1rem 0;'>",
        unsafe_allow_html=True,
    )

    # P2 — Top términos por ciudad
    st.markdown(f"""
    <div class="question-label">P2 — ¿Qué palabras aparecen más en cada ciudad?</div>
    <div class="chart-title">Top términos por ciudad seleccionada</div>
    """, unsafe_allow_html=True)

    col_e2a, col_e2b = st.columns([2, 1])
    with col_e2a:
        ciudad_sel_e2 = st.selectbox(
            "Ciudad", sorted(dff["ciudad"].unique().tolist()), key="ciudad_e2")
    with col_e2b:
        top_n_e2 = st.slider("Top palabras", 5, 20, 12, key="topn_e2")

    wf_city = pd.DataFrame(
        Counter([w for toks in dff[dff["ciudad"] == ciudad_sel_e2]["tokens"]
                 for w in toks]).most_common(top_n_e2),
        columns=["Palabra", "Frecuencia"],
    )
    fig_wc = px.bar(
        wf_city, x="Frecuencia", y="Palabra", orientation="h",
        color="Frecuencia", color_continuous_scale="RdBu_r",
        text="Frecuencia",
    )
    fig_wc.update_traces(textposition="outside", textfont_size=10)
    fig_wc.update_coloraxes(showscale=False)
    fig_wc = apply_theme(fig_wc, 420)
    fig_wc.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        title=f"Palabras más frecuentes — {ciudad_sel_e2}",
    )
    st.plotly_chart(fig_wc, use_container_width=True)

    st.markdown(
        "<hr style='border:none;border-top:1px solid #D9D0C5;margin:1rem 0;'>",
        unsafe_allow_html=True,
    )

    # P4 — Engagement
    st.markdown(f"""
    <div class="question-label">P4 — ¿Qué ciudades generan más debate?</div>
    <div class="chart-title">Comentarios totales y promedio por ciudad</div>
    """, unsafe_allow_html=True)

    debate = (
        dff.groupby("ciudad", dropna=False)
        .agg(
            Publicaciones=("ciudad", "size"),
            Comentarios_totales=("comentarios", "sum"),
            Comentarios_promedio=("comentarios", "mean"),
            Upvotes_totales=("upvotes", "sum"),
        )
        .reset_index()
        .sort_values("Comentarios_totales", ascending=False)
    )

    metrica_debate = st.radio(
        "Métrica de engagement",
        ["Comentarios totales", "Comentarios promedio", "Upvotes totales"],
        horizontal=True, key="metrica_debate",
    )
    col_debate = {
        "Comentarios totales":  "Comentarios_totales",
        "Comentarios promedio": "Comentarios_promedio",
        "Upvotes totales":      "Upvotes_totales",
    }[metrica_debate]

    fig_debate = px.bar(
        debate.sort_values(col_debate, ascending=True),
        x=col_debate, y="ciudad", orientation="h",
        color=col_debate, color_continuous_scale="RdBu",
        text=col_debate,
        labels={col_debate: metrica_debate, "ciudad": "Ciudad"},
    )
    fig_debate.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_debate.update_coloraxes(showscale=False)
    fig_debate = apply_theme(fig_debate, 360)
    st.plotly_chart(fig_debate, use_container_width=True)

    with st.expander("Ver tabla completa de engagement"):
        st.dataframe(
            debate.style.format({
                "Comentarios_totales":  "{:,.0f}",
                "Comentarios_promedio": "{:.1f}",
                "Upvotes_totales":      "{:,.0f}",
            }),
            use_container_width=True, hide_index=True,
        )

# ============================================================
# S — STUDY
# ============================================================
with tab_s:
    st.markdown("""
    <div class="section-intro">
    La fase Study examina las relaciones entre variables: polaridad del sentimiento
    léxico, comportamiento temporal de las publicaciones y comparación multivariada
    del engagement entre ciudades. Esta sección responde las preguntas que requieren
    análisis relacional y temporal del corpus.
    </div>
    """, unsafe_allow_html=True)

    # P3 — Sentimiento
    st.markdown(f"""
    <div class="question-label">P3 — ¿Hay más palabras con polaridad positiva o negativa?</div>
    <div class="chart-title">Análisis de sentimiento léxico por ciudad</div>
    """, unsafe_allow_html=True)

    COLOR_SENT = {"Positivo": C_TEAL_DARK, "Negativo": C_BRICK, "Neutral": C_CREAM}

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        sent_global = dff["sentimiento"].value_counts().reset_index()
        sent_global.columns = ["Sentimiento", "Count"]
        fig_pie = px.pie(
            sent_global, names="Sentimiento", values="Count",
            color="Sentimiento", color_discrete_map=COLOR_SENT, hole=0.45,
        )
        fig_pie = apply_theme(fig_pie, 340)
        fig_pie.update_traces(textposition="outside", textinfo="percent+label")
        fig_pie.update_layout(title="Distribución global de sentimiento", showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_s2:
        sent_city = (
            dff.groupby(["ciudad", "sentimiento"]).size().reset_index(name="n")
        )
        fig_sent = px.bar(
            sent_city, x="ciudad", y="n", color="sentimiento", barmode="stack",
            color_discrete_map=COLOR_SENT,
            labels={"n": "Publicaciones", "ciudad": "Ciudad", "sentimiento": "Sentimiento"},
        )
        fig_sent = apply_theme(fig_sent, 340)
        fig_sent.update_layout(title="Sentimiento por ciudad",
                               xaxis_title="", yaxis_title="Publicaciones")
        st.plotly_chart(fig_sent, use_container_width=True)

    balance = (
        dff.groupby("ciudad")[["pos", "neg"]].sum()
        .assign(balance=lambda x: x["pos"] - x["neg"])
        .reset_index().sort_values("balance", ascending=False)
    )
    fig_balance = px.bar(
        balance, x="ciudad", y="balance",
        color="balance", color_continuous_scale="RdBu_r",
        labels={"balance": "Balance (positivo - negativo)", "ciudad": "Ciudad"},
        text="balance",
    )
    fig_balance.update_traces(texttemplate="%{text:+.0f}", textposition="outside")
    fig_balance = apply_theme(fig_balance, 340)
    fig_balance.update_layout(
        title="Balance de polaridad por ciudad",
        coloraxis_colorbar=dict(title="Balance"),
    )
    fig_balance.add_hline(y=0, line_dash="dash", line_color=COLOR_SUBTEXT, line_width=1)
    st.plotly_chart(fig_balance, use_container_width=True)

    st.markdown(
        "<hr style='border:none;border-top:1px solid #D9D0C5;margin:1rem 0;'>",
        unsafe_allow_html=True,
    )

    # P5 — Evolución temporal
    st.markdown(f"""
    <div class="question-label">P5 — ¿En qué períodos se habla más de cada ciudad?</div>
    <div class="chart-title">Evolución de publicaciones en el tiempo</div>
    """, unsafe_allow_html=True)

    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        cities_temporal = st.multiselect(
            "Ciudades para análisis temporal",
            sorted(dff["ciudad"].unique().tolist()),
            default=sorted(dff["ciudad"].unique().tolist())[:5],
            key="cities_temporal",
        )
    with col_t2:
        granularidad = st.radio("Granularidad", ["Mensual", "Anual"], key="gran")

    dff_time = dff.dropna(subset=["fecha"]).copy()
    if cities_temporal:
        dff_time = dff_time[dff_time["ciudad"].isin(cities_temporal)]
    dff_time["periodo_plot"] = (
        dff_time["fecha"].dt.to_period("M").astype(str)
        if granularidad == "Mensual"
        else dff_time["año"].astype(str)
    )
    serie = (
        dff_time.groupby(["periodo_plot", "ciudad"]).size()
        .reset_index(name="Publicaciones")
    )
    fig_time = px.line(
        serie, x="periodo_plot", y="Publicaciones", color="ciudad",
        color_discrete_sequence=PALETTE_CITIES, markers=True,
        labels={"periodo_plot": "Período", "ciudad": "Ciudad"},
    )
    fig_time = apply_theme(fig_time, 420)
    fig_time.update_layout(xaxis=dict(tickangle=-35),
                           xaxis_title="Período", yaxis_title="Publicaciones")
    st.plotly_chart(fig_time, use_container_width=True)

    st.markdown(
        "<hr style='border:none;border-top:1px solid #D9D0C5;margin:1rem 0;'>",
        unsafe_allow_html=True,
    )

    # P7 — Scatter
    st.markdown(f"""
    <div class="question-label">P7 — ¿Existen diferencias fuertes en engagement entre ciudades?</div>
    <div class="chart-title">Relación entre upvotes y comentarios por publicación</div>
    """, unsafe_allow_html=True)

    max_upvotes = st.slider(
        "Límite de upvotes (para mejor visualización)", 50, 2600, 500, step=50)
    scatter_df = dff.dropna(subset=["upvotes", "comentarios"]).copy()
    scatter_df = scatter_df[scatter_df["upvotes"] <= max_upvotes]

    fig_scatter = px.scatter(
        scatter_df, x="upvotes", y="comentarios",
        color="ciudad", color_discrete_sequence=PALETTE_CITIES, opacity=0.65,
        hover_data={"titulo": True, "subreddit": True},
        labels={"upvotes": "Upvotes", "comentarios": "Comentarios", "ciudad": "Ciudad"},
    )
    fig_scatter = apply_theme(fig_scatter, 460)
    fig_scatter.update_traces(marker_size=6)
    fig_scatter.update_layout(xaxis_title="Upvotes", yaxis_title="Comentarios")
    st.plotly_chart(fig_scatter, use_container_width=True)

# ============================================================
# T — TELL
# ============================================================
with tab_t:
    st.markdown("""
    <div class="section-intro">
    La fase Tell sintetiza los hallazgos del análisis exploratorio y responde
    directamente las preguntas planteadas en la fase Question. Se presentan
    los resultados más relevantes de forma clara y accionable, consolidando
    los patrones identificados a lo largo del proceso de análisis.
    </div>
    """, unsafe_allow_html=True)

    total_pos = int(dff["pos"].sum())
    total_neg = int(dff["neg"].sum())
    ciudad_mas_pubs = dff["ciudad"].value_counts().idxmax() if len(dff) > 0 else "N/A"
    ciudad_mas_debate = (
        dff.groupby("ciudad")["comentarios"].sum().idxmax()
        if dff["comentarios"].notna().any() else "N/A"
    )
    top_subreddit = dff["subreddit"].value_counts().idxmax() if len(dff) > 0 else "N/A"

    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    col_t1.metric("Palabras positivas",  f"{total_pos:,}")
    col_t2.metric("Palabras negativas",  f"{total_neg:,}")
    col_t3.metric("Ciudad más activa",   ciudad_mas_pubs)
    col_t4.metric("Ciudad más debatida", ciudad_mas_debate)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="question-label">Resumen consolidado</div>
    <div class="chart-title">Ranking de ciudades — publicaciones, comentarios y upvotes</div>
    """, unsafe_allow_html=True)

    ranking = (
        dff.groupby("ciudad")
        .agg(Publicaciones=("ciudad", "size"),
             Comentarios=("comentarios", "sum"),
             Upvotes=("upvotes", "sum"))
        .reset_index().sort_values("Publicaciones", ascending=False)
    )
    fig_ranking = go.Figure()
    for met, col in zip(
        ["Publicaciones", "Comentarios", "Upvotes"],
        [C_TEAL_DARK, C_TEAL_LIGHT, C_TERRA_LIGHT],
    ):
        fig_ranking.add_trace(go.Bar(
            name=met, x=ranking["ciudad"], y=ranking[met],
            marker_color=col, opacity=0.88,
        ))
    fig_ranking.update_layout(barmode="group")
    fig_ranking = apply_theme(fig_ranking, 420)
    fig_ranking.update_layout(
        xaxis_title="", yaxis_title="Cantidad",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    st.plotly_chart(fig_ranking, use_container_width=True)

    st.markdown(
        "<hr style='border:none;border-top:1px solid #D9D0C5;margin:1rem 0;'>",
        unsafe_allow_html=True,
    )
    st.markdown(f"""
    <div class="question-label">Respuestas a las preguntas QUEST</div>
    <div class="chart-title">Conclusiones del análisis exploratorio</div>
    """, unsafe_allow_html=True)

    top5_words = [w for w, _ in Counter(
        [word for toks in dff["tokens"] for word in toks]
    ).most_common(5)]

    sent_balance_txt = (
        "predominan las expresiones positivas" if total_pos > total_neg
        else "predominan las expresiones negativas" if total_neg > total_pos
        else "hay un equilibrio entre positivo y negativo"
    )

    periodo_pico_global = "No disponible"
    if not dff.dropna(subset=["fecha"]).empty:
        periodo_pico_global = (
            dff.dropna(subset=["fecha"]).groupby("periodo").size().idxmax()
        )

    respuestas = [
        ("P1", "¿Cuáles son las palabras más frecuentes?",
         f"Los términos con mayor presencia en el corpus son: {', '.join(top5_words)}. "
         "Estos reflejan los temas centrales de conversación en Reddit sobre las ciudades "
         "colombianas y permiten identificar los focos temáticos predominantes en el dataset."),
        ("P2", "¿Qué palabras destacan en cada ciudad?",
         "Cada ciudad presenta un vocabulario particular según sus características sociales, "
         "culturales y coyunturales. El análisis por ciudad en la sección Explore permite "
         "identificar términos diferenciadores que revelan los temas de interés locales."),
        ("P3", "¿Hay más polaridad positiva o negativa?",
         f"En el corpus filtrado, {sent_balance_txt} "
         f"(positivas: {total_pos:,} — negativas: {total_neg:,}). "
         "La distribución varía según la ciudad y el período, reflejando diferentes "
         "percepciones urbanas entre los usuarios de la plataforma."),
        ("P4", "¿Qué ciudades generan más debate?",
         f"La ciudad con mayor volumen de comentarios es {ciudad_mas_debate}, "
         "lo que indica que sus publicaciones generan mayor interacción y discusión. "
         "El engagement no siempre es proporcional al número de publicaciones totales."),
        ("P5", "¿En qué períodos se habla más de cada ciudad?",
         f"El período con mayor actividad global en los datos filtrados es {periodo_pico_global}. "
         "La evolución temporal en la sección Study permite identificar picos por ciudad "
         "y relacionarlos con eventos locales o coyunturas específicas del momento."),
        ("P6", "¿Qué subreddits concentran la discusión?",
         f"El subreddit con mayor participación es r/{top_subreddit}. "
         "La distribución por foro varía según la ciudad: algunas cuentan con subreddits "
         "dedicados, mientras que otras se discuten principalmente en r/Colombia."),
        ("P7", "¿Existen diferencias fuertes entre ciudades?",
         "Sí. Bogotá y Medellín concentran el mayor volumen de publicaciones y engagement, "
         "mientras que ciudades como Pereira y Bucaramanga tienen presencia significativamente "
         "menor. La relación entre upvotes y comentarios es positiva pero moderada, "
         "con alta variabilidad entre publicaciones individuales."),
    ]

    for code, pregunta, respuesta in respuestas:
        col_ra, col_rb = st.columns([0.08, 0.92])
        with col_ra:
            st.markdown(f"""
            <div style="font-family:'Playfair Display',serif;font-size:1rem;
            color:{C_TEAL_DARK};font-weight:700;margin-top:0.9rem;">{code}</div>
            """, unsafe_allow_html=True)
        with col_rb:
            st.markdown(f"""
            <div style="margin-top:0.9rem;">
              <div class="pregunta-txt">{pregunta}</div>
              <div class="respuesta-txt">{respuesta}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(
            "<hr style='border:none;border-top:1px solid #EDE7DC;margin:0.5rem 0;'>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="question-label">Períodos pico por ciudad</div>
    <div class="chart-title">Mes con mayor actividad para cada ciudad</div>
    """, unsafe_allow_html=True)

    pico_por_ciudad = (
        dff.dropna(subset=["fecha"])
        .groupby(["ciudad", "periodo"]).size()
        .reset_index(name="publicaciones")
        .sort_values("publicaciones", ascending=False)
        .groupby("ciudad").first()
        .reset_index()
        .sort_values("publicaciones", ascending=False)
    )
    st.dataframe(
        pico_por_ciudad[["ciudad", "periodo", "publicaciones"]].rename(columns={
            "ciudad":        "Ciudad",
            "periodo":       "Período pico",
            "publicaciones": "Publicaciones en el pico",
        }),
        use_container_width=True, hide_index=True,
    )

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family:'DM Sans',sans-serif;font-size:0.78rem;
    color:#9A8F85;text-align:center;padding:1rem 0;
    border-top:1px solid #D9D0C5;">
    EDA · Ciudades de Colombia · Framework QUEST · Grupo 2 · Modelos Analíticos
    </div>
    """, unsafe_allow_html=True)
