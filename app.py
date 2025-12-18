import streamlit as st
import pandas as pd
import base64
from model import df, fix_unhashable, clean_coordinates, recommend_hybrid

# ----------------------------- 
# Page config & session state
# ----------------------------- 
st.set_page_config(
    page_title="Wisata Lampung Recommender",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if "page" not in st.session_state:
    st.session_state.page = "home"
if "selected_place" not in st.session_state:
    st.session_state.selected_place = None

# ----------------------------- 
# Load & prepare data
# ----------------------------- 
df_clean = clean_coordinates(fix_unhashable(df))
PLACEHOLDER = "https://via.placeholder.com/400x250?text=No+Image"

if "Foto" not in df_clean.columns:
    df_clean["Foto"] = PLACEHOLDER
df_clean["Foto"] = df_clean["Foto"].apply(
    lambda x: x.strip() if isinstance(x, str) and x.strip() != "" else PLACEHOLDER
)

ALPHA1, ALPHA2, BETA = 0.5, 0.3, 0.2

# ----------------------------- 
# Modern Lampung-themed CSS (Biru Tosca & Gold)
# ----------------------------- 
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Lampung-themed Background - Biru Gelap Tosca dengan Aksen Gold */
    .stApp {
        background: linear-gradient(135deg, #0d3b52 0%, #1a5266 50%, #266b7a 100%);
        background-attachment: fixed;
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: 
            radial-gradient(circle at 20% 80%, rgba(212, 175, 55, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 215, 0, 0.06) 0%, transparent 50%),
            url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 80"><path d="M14 16H9v-2h5V9h2v5h5v2h-5v5h-2v-5zM40 40h-5v-2h5v-5h2v5h5v2h-5v5h-2v-5z" fill="rgba(212,175,55,0.04)"/></svg>');
        pointer-events: none;
        z-index: 0;
    }
    
    .main > div {
        position: relative;
        z-index: 1;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hero Section - Gold & Tosca Lampung Style */
    .hero-section {
        background: linear-gradient(135deg, #1a5266 0%, #266b7a 50%, #2d7e8c 100%);
        padding: 3.5rem 2rem;
        border-radius: 24px;
        text-align: center;
        color: white;
        margin-bottom: 2.5rem;
        box-shadow: 0 20px 60px rgba(26, 82, 102, 0.5);
        position: relative;
        overflow: hidden;
        border: 3px solid rgba(212, 175, 55, 0.3);
    }
    
    .hero-section::before {
        content: '🌴';
        position: absolute;
        font-size: 150px;
        opacity: 0.08;
        left: -30px;
        top: -30px;
        transform: rotate(-15deg);
    }
    
    .hero-section::after {
        content: '🏖️';
        position: absolute;
        font-size: 120px;
        opacity: 0.08;
        right: -20px;
        bottom: -20px;
        transform: rotate(15deg);
    }
    
    .hero-logo {
        width: 110px;
        height: 110px;
        border-radius: 50%;
        margin-bottom: 1.2rem;
        border: 4px solid rgba(212, 175, 55, 0.6);
        box-shadow: 0 10px 40px rgba(212, 175, 55, 0.3);
        position: relative;
        z-index: 2;
    }
    
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        margin-bottom: 0.8rem;
        color: #ffd700;
        text-shadow: 2px 4px 8px rgba(0, 0, 0, 0.4);
        position: relative;
        z-index: 2;
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        font-size: 1.4rem;
        font-weight: 500;
        margin-bottom: 0.8rem;
        position: relative;
        z-index: 2;
        color: #f5f5f5;
        text-shadow: 1px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .hero-description {
        font-size: 1.05rem;
        opacity: 0.95;
        font-weight: 400;
        position: relative;
        z-index: 2;
        color: #e8e8e8;
    }
    
    /* Stats Bar */
    .stats-bar {
        display: flex;
        justify-content: space-around;
        background: rgba(255, 255, 255, 0.98);
        padding: 1.8rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        border: 2px solid rgba(212, 175, 55, 0.3);
    }
    
    .stat-item {
        text-align: center;
        flex: 1;
    }
    
    .stat-number {
        font-size: 2.2rem;
        font-weight: 800;
        color: #d4af37;
        margin-bottom: 0.3rem;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    .stat-label {
        font-size: 0.95rem;
        color: #1a5266;
        font-weight: 600;
    }
    
    /* Filter Section */
    .filter-section {
        background: rgba(255, 255, 255, 0.98);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        margin-bottom: 2.5rem;
        border: 2px solid rgba(212, 175, 55, 0.3);
    }
    
    .filter-title {
        font-size: 1.9rem;
        font-weight: 700;
        color: #0d3b52;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .filter-subtitle {
        color: #2d3748;
        font-size: 1.05rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Streamlit Elements Styling */
    .stSelectbox > div > div {
        background-color: #ffffff;
        border: 2px solid #266b7a;
        border-radius: 12px;
        color: #0d3b52;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #d4af37;
        box-shadow: 0 4px 12px rgba(212, 175, 55, 0.3);
    }
    
    .stSelectbox label {
        color: #d4af37;
        font-weight: 700;
        font-size: 1.1rem;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
    }
    
    /* Styling untuk dropdown options */
    .stSelectbox [data-baseweb="select"] {
        color: #0d3b52;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        background-color: #ffffff;
        color: #0d3b52;
        font-weight: 600;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #d4af37 0%, #f4d03f 100%);
        color: #0d3b52;
        border: none;
        border-radius: 12px;
        padding: 0.9rem 2rem;
        font-size: 1.1rem;
        font-weight: 700;
        box-shadow: 0 6px 20px rgba(212, 175, 55, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(212, 175, 55, 0.6);
        background: linear-gradient(135deg, #f4d03f 0%, #d4af37 100%);
    }
    
    /* Detail Page */
    .detail-container {
        background: rgba(255, 255, 255, 0.98);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
        border: 2px solid rgba(212, 175, 55, 0.3);
    }
    
    .detail-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1a5266 0%, #266b7a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
        line-height: 1.2;
    }
    
    .detail-image-container {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
        border: 3px solid rgba(212, 175, 55, 0.4);
    }
    
    .info-card {
        background: linear-gradient(135deg, #1a5266 0%, #266b7a 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        height: 100%;
        box-shadow: 0 10px 40px rgba(26, 82, 102, 0.4);
        border: 2px solid rgba(212, 175, 55, 0.3);
    }
    
    .info-item {
        display: flex;
        align-items: center;
        margin-bottom: 1.3rem;
        padding: 1.2rem;
        background: rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .info-icon {
        font-size: 1.8rem;
        margin-right: 1rem;
    }
    
    .info-content {
        flex: 1;
    }
    
    .info-label {
        font-size: 0.85rem;
        opacity: 0.9;
        font-weight: 400;
        margin-bottom: 0.3rem;
        color: #f0f0f0;
    }
    
    .info-value {
        font-size: 1.15rem;
        font-weight: 600;
        color: #ffffff;
    }
    
    /* Recommendations Section */
    .rec-section {
        background: rgba(255, 255, 255, 0.98);
        padding: 2.5rem;
        border-radius: 20px;
        margin-top: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
        border: 2px solid rgba(212, 175, 55, 0.3);
    }
    
    .rec-header {
        text-align: center;
        margin-bottom: 2.5rem;
    }
    
    .rec-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1a5266 0%, #266b7a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .rec-subtitle {
        color: #2d3748;
        font-size: 1.05rem;
        font-weight: 500;
    }
    
    /* Grid Layout for Recommendations */
    .rec-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 2rem;
        margin-top: 2rem;
    }
    
    .rec-card {
        background: white;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        border: 2px solid transparent;
    }
    
    .rec-card:hover {
        transform: translateY(-12px);
        box-shadow: 0 20px 50px rgba(26, 82, 102, 0.3);
        border-color: #d4af37;
    }
    
    .rec-card-image-container {
        position: relative;
        overflow: hidden;
        height: 220px;
    }
    
    .rec-card-image {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.4s ease;
    }
    
    .rec-card:hover .rec-card-image {
        transform: scale(1.1);
    }
    
    .rec-card-badge {
        position: absolute;
        top: 12px;
        right: 12px;
        background: linear-gradient(135deg, #d4af37 0%, #f4d03f 100%);
        color: #0d3b52;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .rec-card-content {
        padding: 1.5rem;
    }
    
    .rec-card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0d3b52;
        margin-bottom: 1rem;
        line-height: 1.3;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    .rec-card-info {
        display: flex;
        flex-direction: column;
        gap: 0.6rem;
    }
    
    .rec-info-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #1a202c;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    .rec-info-icon {
        color: #d4af37;
    }
    
    /* Back Button */
    .back-btn-container {
        position: fixed;
        top: 1.5rem;
        left: 1.5rem;
        z-index: 1000;
    }
    
    .back-btn {
        background: white;
        color: #1a5266;
        border: 2px solid #1a5266;
        border-radius: 30px;
        padding: 0.8rem 1.8rem;
        font-size: 1rem;
        font-weight: 600;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
    }
    
    .back-btn:hover {
        background: #1a5266;
        color: white;
        transform: translateX(-5px);
        box-shadow: 0 8px 30px rgba(26, 82, 102, 0.4);
    }
    
    /* Popular Destinations Section */
    .popular-section {
        background: rgba(255, 255, 255, 0.98);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        border: 2px solid rgba(212, 175, 55, 0.3);
    }
    
    .section-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1a5266 0%, #266b7a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    /* Styling untuk card titles dan captions di popular section */
    .popular-section .stMarkdown {
        color: #0d3b52;
    }
    
    .popular-section strong {
        color: #0d3b52;
        font-weight: 700;
    }
    
    .popular-section .stCaption {
        color: #1a5266 !important;
        font-weight: 600;
    }
    
    /* Override Streamlit default text colors */
    .stMarkdown p {
        color: inherit;
    }
    
    .element-container strong {
        color: #0d3b52;
    }
    
    /* Loading Animation */
    .loading {
        text-align: center;
        padding: 3rem;
        color: #d4af37;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.2rem;
        }
        
        .stats-bar {
            flex-direction: column;
            gap: 1rem;
        }
        
        .rec-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------- 
# Helper Functions
# ----------------------------- 
def load_image_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

def show_hero():
    try:
        logo64 = load_image_base64("sainsdataaa.jpg")
        logo_src = f"data:image/png;base64,{logo64}" if logo64 else PLACEHOLDER
    except:
        logo_src = PLACEHOLDER
    
    st.markdown(f"""
    <div class="hero-section">
        <img src="{logo_src}" class="hero-logo" alt="Logo Lampung">
        <div class="hero-title">🌴 Jelajahi Lampung</div>
        <div class="hero-subtitle">Sistem Rekomendasi Wisata Cerdas</div>
        <div class="hero-description">Project Sains Data Kelompok 2 — Temukan destinasi impian Anda dengan teknologi AI</div>
    </div>
    """, unsafe_allow_html=True)

def show_stats():
    total_destinations = len(df_clean)
    total_categories = df_clean['kategori'].nunique()
    total_cities = df_clean['kabupaten kota'].nunique()
    avg_rating = df_clean['rating'].mean()
    
    st.markdown(f"""
    <div class="stats-bar">
        <div class="stat-item">
            <div class="stat-number">{total_destinations}</div>
            <div class="stat-label">🏖️ Destinasi Wisata</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">{total_categories}</div>
            <div class="stat-label">🎯 Kategori</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">{total_cities}</div>
            <div class="stat-label">📍 Kabupaten/Kota</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">{avg_rating:.1f}</div>
            <div class="stat-label">⭐ Rating Rata-rata</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def go_home():
    st.session_state.page = "home"
    st.session_state.selected_place = None

def go_detail(place):
    st.session_state.selected_place = place
    st.session_state.page = "detail"

# ----------------------------- 
# Page: Home
# ----------------------------- 
def page_home():
    show_hero()
    show_stats()
    
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    st.markdown("""
    <div class="filter-title">🔍 Mulai Petualangan Anda</div>
    <div class="filter-subtitle">Pilih preferensi wisata dan temukan destinasi terbaik di Lampung</div>
    """, unsafe_allow_html=True)
    
    # Filter Section
    col1, col2 = st.columns(2)
    kab_list = ["Semua Kabupaten/Kota"] + sorted(df_clean["kabupaten kota"].unique())
    selected_kab = col1.selectbox("📍 Pilih Lokasi", kab_list, key="kab_select")
    
    kat_list = ["Semua Kategori"] + sorted(df_clean["kategori"].unique())
    selected_cat = col2.selectbox("🎯 Pilih Kategori Wisata", kat_list, key="cat_select")
    
    # Filter data
    df_filtered = df_clean.copy()
    if selected_kab != "Semua Kabupaten/Kota":
        df_filtered = df_filtered[df_filtered["kabupaten kota"] == selected_kab]
    if selected_cat != "Semua Kategori":
        df_filtered = df_filtered[df_filtered["kategori"] == selected_cat]
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Destination Selection
    selected_place = st.selectbox(
        "🏖️ Pilih Destinasi Wisata",
        df_filtered["Nama tempat"].tolist(),
        key="place_select"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Eksplor Destinasi & Dapatkan Rekomendasi", key="explore_btn", use_container_width=True):
        go_detail(selected_place)
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Popular Destinations Preview
    st.markdown('<div class="popular-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔥 Destinasi Populer Lampung</div>', unsafe_allow_html=True)
    
    popular = df_clean.nlargest(6, 'rating')
    
    cols = st.columns(3)
    for idx, (i, row) in enumerate(popular.iterrows()):
        with cols[idx % 3]:
            st.image(row['Foto'], use_container_width=True)
            st.markdown(f"**{row['Nama tempat']}**")
            st.caption(f"⭐ {row['rating']} | 📍 {row['kabupaten kota']}")
            if st.button("Lihat Detail", key=f"pop_{idx}", use_container_width=True):
                go_detail(row['Nama tempat'])
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------- 
# Page: Detail
# ----------------------------- 
def page_detail():
    # Back Button
    st.markdown('<div class="back-btn-container">', unsafe_allow_html=True)
    if st.button("⬅ Kembali", key="back_btn"):
        go_home()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    place = st.session_state.selected_place
    if place is None:
        go_home()
        st.rerun()
        return
    
    # Get place details
    row = df_clean[df_clean["Nama tempat"] == place].iloc[0]
    
    st.markdown('<div class="detail-container">', unsafe_allow_html=True)
    st.markdown(f'<div class="detail-title">{place}</div>', unsafe_allow_html=True)
    
    colA, colB = st.columns([1.4, 1.6])
    
    with colA:
        st.markdown('<div class="detail-image-container">', unsafe_allow_html=True)
        st.image(row["Foto"], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with colB:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-item">
                <div class="info-icon">📂</div>
                <div class="info-content">
                    <div class="info-label">Kategori Wisata</div>
                    <div class="info-value">{row['kategori']}</div>
                </div>
            </div>
            <div class="info-item">
                <div class="info-icon">📍</div>
                <div class="info-content">
                    <div class="info-label">Lokasi</div>
                    <div class="info-value">{row['kabupaten kota']}</div>
                </div>
            </div>
            <div class="info-item">
                <div class="info-icon">🏗️</div>
                <div class="info-content">
                    <div class="info-label">Fasilitas Tersedia</div>
                    <div class="info-value">{row['Fasilitas']}</div>
                </div>
            </div>
            <div class="info-item">
                <div class="info-icon">⭐</div>
                <div class="info-content">
                    <div class="info-label">Rating Pengunjung</div>
                    <div class="info-value">{row['rating']} / 5.0</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Recommendations Section
    st.markdown("""
    <div class="rec-section">
        <div class="rec-header">
            <div class="rec-title">✨ Rekomendasi Destinasi Serupa</div>
            <div class="rec-subtitle">Berdasarkan analisis AI, berikut destinasi wisata yang cocok untuk Anda</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    rec = recommend_hybrid(
        df_clean, target=place, top_n=9, by_name=True,
        alpha1=ALPHA1, alpha2=ALPHA2, beta=BETA
    )
    
    # Create grid layout (3 columns per row)
    num_cols = 3
    rows = (len(rec) + num_cols - 1) // num_cols
    
    for row_idx in range(rows):
        cols = st.columns(num_cols)
        for col_idx in range(num_cols):
            idx = row_idx * num_cols + col_idx
            if idx < len(rec):
                r = rec.iloc[idx]
                foto = r["Foto"] if isinstance(r["Foto"], str) else PLACEHOLDER
                
                with cols[col_idx]:
                    with st.container():
                        st.markdown('<div class="rec-card">', unsafe_allow_html=True)
                        
                        # Image with badge
                        st.markdown(f"""
                        <div class="rec-card-image-container">
                            <img src="{foto}" class="rec-card-image" alt="{r['Nama tempat']}">
                            <div class="rec-card-badge">⭐ {r['score_hybrid']:.3f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Content
                        st.markdown(f"""
                        <div class="rec-card-content">
                            <div class="rec-card-title">{r['Nama tempat']}</div>
                            <div class="rec-card-info">
                                <div class="rec-info-item">
                                    <span class="rec-info-icon">📂</span>
                                    <span>{r['kategori']}</span>
                                </div>
                                <div class="rec-info-item">
                                    <span class="rec-info-icon">📏</span>
                                    <span>{r['distance_km']:.2f} km dari sini</span>
                                </div>
                                <div class="rec-info-item">
                                    <span class="rec-info-icon">📍</span>
                                    <span>{r['kabupaten kota']}</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Button
                        if st.button("👉 Lihat Detail", key=f"rec_{idx}", use_container_width=True):
                            go_detail(r['Nama tempat'])
                            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------- 
# Main App
# ----------------------------- 
if st.session_state.page == "home":
    page_home()
else:
    page_detail()