import streamlit as st
import pandas as pd
import plotly.express as px

from ml_core import load_model

st.set_page_config(
    page_title="News Analytics Dashboard",
    page_icon="📰",
    layout="wide"
)

# ---------- STYLE ----------
st.markdown("""
<style>
/* ---------- Base ---------- */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top left, rgba(250,244,205,0.35), transparent 28%),
        radial-gradient(circle at top right, rgba(255,255,255,0.28), transparent 24%),
        linear-gradient(180deg, #93ABD8 0%, #9ab1db 100%);
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #6f89bf 0%, #7f99cc 100%);
    border-right: 1px solid rgba(255,255,255,0.18);
}

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
    padding-left: 2.2rem;
    padding-right: 2.2rem;
    max-width: 1500px;
}

/* ---------- Typography ---------- */
html, body, [class*="css"]  {
    color: #0f172a;
}

h1, h2, h3, h4 {
    color: #0f172a !important;
    letter-spacing: -0.02em;
}

p, span, label, div {
    color: #1e293b;
}

/* ---------- Hero ---------- */
.hero {
    position: relative;
    overflow: hidden;
    padding: 30px 34px;
    border-radius: 28px;
    background:
        linear-gradient(135deg, rgba(250,244,205,0.78), rgba(255,255,255,0.58));
    border: 1px solid rgba(255,255,255,0.55);
    box-shadow:
        0 20px 40px rgba(15, 23, 42, 0.10),
        inset 0 1px 0 rgba(255,255,255,0.5);
    margin-bottom: 20px;
}

.hero::after {
    content: "";
    position: absolute;
    right: -40px;
    top: -40px;
    width: 180px;
    height: 180px;
    background: radial-gradient(circle, rgba(63,90,166,0.18), transparent 70%);
    border-radius: 50%;
}

.hero-title {
    font-size: 50px;
    font-weight: 800;
    color: #0f172a;
    margin: 8px 0 8px 0;
    line-height: 1.02;
}

.hero-subtitle {
    font-size: 17px;
    color: #334155;
    max-width: 940px;
    line-height: 1.55;
    margin-bottom: 0;
}

.badges-wrap {
    margin-bottom: 8px;
}

.badge {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(63,90,166,0.12);
    color: #244089;
    border: 1px solid rgba(63,90,166,0.18);
    font-size: 12px;
    font-weight: 700;
    margin-right: 8px;
    margin-bottom: 8px;
}

/* ---------- Cards ---------- */
.metric-card {
    background: rgba(255,255,255,0.78);
    backdrop-filter: blur(8px);
    border-radius: 22px;
    padding: 18px 20px;
    border: 1px solid rgba(255,255,255,0.6);
    box-shadow: 0 12px 26px rgba(15,23,42,0.08);
    min-height: 120px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 16px 32px rgba(15,23,42,0.12);
}

.metric-label {
    color: #475569;
    font-size: 14px;
    margin-bottom: 8px;
    font-weight: 600;
}

.metric-value {
    color: #0f172a;
    font-size: 31px;
    font-weight: 800;
    line-height: 1.1;
}

.metric-sub {
    color: #475569;
    font-size: 13px;
    margin-top: 8px;
    line-height: 1.4;
}

.section-card {
    background: rgba(255,255,255,0.80);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.62);
    border-radius: 24px;
    padding: 22px 22px 16px 22px;
    box-shadow: 0 14px 32px rgba(15,23,42,0.08);
    margin-top: 8px;
    margin-bottom: 16px;
}

.muted {
    color: #475569;
    font-size: 14px;
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 14px;
    margin-top: 6px;
    margin-bottom: 10px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.40);
    color: #0f172a;
    border-radius: 14px;
    padding: 10px 18px;
    border: 1px solid rgba(255,255,255,0.55);
    font-weight: 700;
    box-shadow: 0 6px 12px rgba(15,23,42,0.04);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3F5AA6, #5e79c4);
    color: white !important;
}

/* ---------- Inputs ---------- */
div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.55);
    border-radius: 18px;
    padding: 10px;
    border: 1px solid rgba(255,255,255,0.7);
}

.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(135deg, #3F5AA6, #5672bf);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    padding: 0.55rem 1rem;
    box-shadow: 0 8px 18px rgba(63,90,166,0.22);
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #334b8c, #4f67a8);
    color: white;
}

/* ---------- Tables ---------- */
[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid rgba(15,23,42,0.06);
    box-shadow: 0 10px 20px rgba(15,23,42,0.05);
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] * {
    color: #FAF4CD !important;
}

.sidebar-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.14);
    padding: 14px;
    border-radius: 18px;
    margin-bottom: 12px;
}

.footer-note {
    color: #334155;
    opacity: 0.85;
    font-size: 13px;
    text-align: center;
    margin-top: 10px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------- MODEL ----------
model, encoder = None, None
model_loaded = False
try:
    model, encoder = load_model()
    model_loaded = True
except Exception:
    model_loaded = False

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## ⚙️ Навигация")
    st.markdown("""
    <div class="sidebar-card">
        <b>О проекте</b><br><br>
        Веб-сервис для автоматической классификации новостных текстов
        по тематическим категориям.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <b>Технологии</b><br><br>
        • Python<br>
        • Streamlit<br>
        • TF-IDF<br>
        • SVM<br>
        • Pandas / Scikit-learn
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <b>Формат данных</b><br><br>
        Для классификации нужен CSV со столбцом <code>text</code>.<br>
        Для аналитики нужен CSV со столбцом <code>category</code>.
    </div>
    """, unsafe_allow_html=True)

    if model_loaded:
        st.success("Модель загружена")
    else:
        st.error("Модель не загружена")

# ---------- HERO ----------
st.markdown("""
<div class="hero">
    <div class="badges-wrap">
        <span class="badge">ML</span>
        <span class="badge">NLP</span>
        <span class="badge">Dashboard</span>
        <span class="badge">Streamlit</span>
    </div>
    <div class="hero-title">📰 Классификация новостей</div>
    <p class="hero-subtitle">
        Интерактивный дашборд для загрузки, автоматической классификации и аналитики
        новостных текстов по тематическим категориям. Сервис использует методы обработки
        естественного языка и машинного обучения для быстрого анализа новостных материалов.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------- KPI ----------
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Модель</div>
        <div class="metric-value">SVM</div>
        <div class="metric-sub">Классическая модель для устойчивой текстовой классификации</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Точность</div>
        <div class="metric-value">0.86</div>
        <div class="metric-sub">Результат на подготовленном датасете новостей</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Формат входа</div>
        <div class="metric-value">CSV</div>
        <div class="metric-sub">Минимальное требование: наличие столбца <b>text</b></div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    system_status = "Готова" if model_loaded else "Нет"
    system_sub = "Можно запускать анализ" if model_loaded else "Сначала обучи модель"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Система</div>
        <div class="metric-value">{system_status}</div>
        <div class="metric-sub">{system_sub}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# ---------- TABS ----------
tab1, tab2 = st.tabs(["📂 Классификация", "📊 Аналитика"])

# ---------- CLASSIFICATION ----------
with tab1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Загрузка и классификация CSV")
    st.markdown(
        '<div class="muted">Загрузите CSV-файл со столбцом <b>text</b>. '
        'Система автоматически определит категорию каждой новости и сформирует итоговую таблицу.</div>',
        unsafe_allow_html=True
    )

    file = st.file_uploader("Выберите файл для классификации", type=["csv"])

    if file is not None:
        if not model_loaded:
            st.warning("Модель не загружена. Сначала обучите модель.")
        else:
            try:
                df = pd.read_csv(file)

                if "text" not in df.columns:
                    st.error("Файл должен содержать столбец text")
                else:
                    preds = model.predict(df["text"])
                    df["category"] = encoder.inverse_transform(preds)

                    st.success("Классификация выполнена успешно")

                    top_left, top_right = st.columns([3, 1])
                    with top_left:
                        st.markdown("### Результаты")
                    with top_right:
                        st.markdown(
                            f'<div class="muted" style="text-align:right; padding-top:10px;">'
                            f'Записей: <b>{len(df)}</b></div>',
                            unsafe_allow_html=True
                        )

                    filter_options = ["Все"] + sorted(df["category"].dropna().unique().tolist())
                    selected_category = st.selectbox(
                        "Фильтр по категории",
                        filter_options,
                        index=0
                    )

                    display_df = df.copy()
                    if selected_category != "Все":
                        display_df = display_df[display_df["category"] == selected_category]

                    st.dataframe(display_df, use_container_width=True, height=430)

                    csv_result = display_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "📥 Скачать результат",
                        csv_result,
                        "classified_news.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Ошибка при обработке файла: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- ANALYTICS ----------
with tab2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Аналитика категорий")
    st.markdown(
        '<div class="muted">Загрузите CSV со столбцом <b>category</b>, чтобы построить распределение '
        'новостей по темам и оценить структуру выборки.</div>',
        unsafe_allow_html=True
    )

    analytics_file = st.file_uploader(
        "Выберите файл для аналитики",
        type=["csv"],
        key="analytics_uploader"
    )

    if analytics_file is not None:
        try:
            df = pd.read_csv(analytics_file)

            if "category" not in df.columns:
                st.error("Файл должен содержать столбец category")
            else:
                counts = df["category"].value_counts().reset_index()
                counts.columns = ["category", "count"]

                total_news = int(counts["count"].sum()) if not counts.empty else 0
                total_categories = len(counts)
                top_category = counts.iloc[0]["category"] if not counts.empty else "-"
                avg_per_category = int(total_news / total_categories) if total_categories else 0

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Всего новостей", total_news)
                with m2:
                    st.metric("Категорий", total_categories)
                with m3:
                    st.metric("Топ-категория", top_category)
                with m4:
                    st.metric("Среднее на категорию", avg_per_category)

                left, right = st.columns(2)

                with left:
                    fig_bar = px.bar(
                        counts,
                        x="category",
                        y="count",
                        title="Количество новостей по категориям",
                        text_auto=True
                    )
                    fig_bar.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(255,255,255,0.55)",
                        font_color="#0f172a",
                        xaxis_title="Категория",
                        yaxis_title="Количество"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                with right:
                    fig_pie = px.pie(
                        counts,
                        names="category",
                        values="count",
                        title="Распределение категорий"
                    )
                    fig_pie.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#0f172a"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                st.markdown("### Таблица распределения")
                st.dataframe(counts, use_container_width=True, height=360)

        except Exception as e:
            st.error(f"Ошибка при построении аналитики: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer-note">Демонстрационная версия системы классификации новостей</div>',
    unsafe_allow_html=True
)