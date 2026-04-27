import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Konfigurasi Halaman

st.set_page_config(
    page_title="Prediksi Best Seller",
    page_icon="🛍️",
    layout="wide"  
)


# Load model + Error handling
@st.cache_resource
def load_model():
    try:
        with open("model_bestseller.pickle", "rb") as file:
           return pickle.load(file)
    except FileNotFoundError:
        return None

model_package = load_model()

# Validasi model berhasil di load
if model_package is None:
    st.error("File 'model_bestseller.pickle' tidak ditemukan! Pastikan kamu sudah jalankan main.ipynb sampai selesai.")
    st.stop()

# Menarik variabel dari package (SINKRONISASI DENGAN NOTEBOOK)
model = model_package['model']      
FEATURES = model_package['features']
THRESHOLD = model_package['threshold']
STATS = model_package['market_stats']
IMPORTANCES = STATS['feature_importances'] 


def buat_input(harga, diskon, rating, ulasan):
    harga_efektif = harga * (1 - diskon / 100)
    return pd.DataFrame([{
        'Harga (IDR)': harga,
        'Diskon (%)': diskon,
        'Rating': rating,
        'Ulasan_bersih': ulasan,
        'Harga_setelah_diskon': harga_efektif,
        'Ada_diskon': 1 if diskon > 0 else 0,
        'Skor_kepercayaan': rating * ulasan,
    }])[FEATURES]

def prediksi(harga, diskon, rating, ulasan):
    data = buat_input(harga, diskon, rating, ulasan)
    kelas = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1] * 100
    return kelas, prob

def gauge_chart(probabilitas: float) -> str:
    import math

    # Tambahkan nilai default untuk zona di paling atas
    zona = "Kurang Berpotensi" 
    
    if probabilitas < 40:
        color = "#ef4444"
        zona = "Risiko Tinggi"
    elif probabilitas < 70:
        color = "#f59e0b"
        zona = "Cukup Berpotensi"
    else:
        color = "#22c55e"
        zona = "Sangat Berpotensi"
    
    angle_deg = probabilitas * 180 / 100
    angle_rad = math.radians(180 - angle_deg) # 180 - supaya jarum bergerak dari kiri ke kanan

    cx, cy, r = 150, 130, 100
    needle_x = cx + r * math.cos(angle_rad)
    needle_y = cy - r * math.sin(angle_rad)

    svg = f"""
    <svg viewBox="0 0 300 160" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:320px">
      <!-- Background arc (abu-abu) -->
      <path d="M 50 130 A 100 100 0 0 1 250 130"
            fill="none" stroke="#e5e7eb" stroke-width="18" stroke-linecap="round"/>
      <!-- Colored arc (progress) -->
      <path d="M 50 130 A 100 100 0 0 1 {needle_x:.1f} {needle_y:.1f}"
            fill="none" stroke="{color}" stroke-width="18" stroke-linecap="round"/>
      <!-- Jarum -->
      <line x1="{cx}" y1="{cy}" x2="{needle_x:.1f}" y2="{needle_y:.1f}"
            stroke="#374151" stroke-width="3" stroke-linecap="round"/>
      <circle cx="{cx}" cy="{cy}" r="6" fill="#374151"/>
      <!-- Label angka -->
      <text x="{cx}" y="{cy + 28}" text-anchor="middle"
            font-size="26" font-weight="bold" fill="{color}">{probabilitas:.1f}%</text>
      <!-- Label zona -->
      <text x="{cx}" y="{cy + 50}" text-anchor="middle"
            font-size="11" fill="#6b7280">{zona}</text>
      <!-- Label 0% dan 100% -->
      <text x="44"  y="148" text-anchor="middle" font-size="10" fill="#9ca3af">0%</text>
      <text x="256" y="148" text-anchor="middle" font-size="10" fill="#9ca3af">100%</text>
    </svg>
    """
    return svg

def tampilkan_benchmark(harga, diskon, rating, ulasan):
    st.markdown("### 📊 Posisi Produkmu vs Pasar")

    def buat_baris(label, nilai_user, nilai_median, satuan="", format_rp=False):
        if format_rp:
            u = f"Rp{nilai_user:,.0f}"
            m = f"Rp{nilai_median:,.0f}"
        else:
            u = f"{nilai_user}{satuan}"
            m = f"{nilai_median}{satuan}"
        
        delta = nilai_user - nilai_median
        if format_rp:
            delta_pct = (delta / nilai_median *100) if nilai_median > 0 else 0
            delta_str = f"+{delta_pct:0f}%" if delta > 0 else f"{delta_pct:0f}%"
        else:
            delta_str = f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}"
        
        arah = "⬆️" if delta > 0 else "⬇️" if delta < 0 else "➡️"
        return {"Metrik": label, "Produkmu": u, "Median Pasar": m, "Selisih": f"{arah} {delta_str}"}

    rows = [
        buat_baris("Harga",  harga,  STATS['median_harga'],  format_rp=True),
        buat_baris("Diskon", diskon, STATS['median_diskon'],  satuan="%"),
        buat_baris("Rating", rating, STATS['median_rating']),
        buat_baris("Ulasan", ulasan, STATS['median_ulasan'],  satuan=" ulasan"),
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def tampilkan_feature_importance():
    label_map = {
        'Harga (IDR)':          '💰 Harga Asli',
        'Diskon (%)':           '🏷️ Diskon',
        'Rating':               '⭐ Rating',
        'Ulasan_bersih':        '💬 Jumlah Ulasan',
        'Harga_setelah_diskon': '💵 Harga Setelah Diskon',
        'Ada_diskon':           '✅ Ada/Tidak Diskon',
        'Skor_kepercayaan':     '🤝 Skor Kepercayaan',
    }
    data = {label_map.get(k, k): v for k, v in IMPORTANCES.items()}
    df_imp = pd.DataFrame.from_dict(data, orient='index', columns=['Importance'])
    df_imp = df_imp.sort_values('Importance', ascending=True)
    st.bar_chart(df_imp)

def tampilkan_saran(prediksi_kelas, prob, harga, diskon, rating, ulasan):
    saran = []
    if diskon == 0:
        saran.append("🏷️ **Tambahkan diskon** minimal 10–20% untuk menarik perhatian pembeli.")
    if harga > 500_000 and diskon < 20:
        saran.append("💰 Harga di atas Rp 500rb butuh **diskon lebih besar** agar kompetitif.")
    if 0 < rating < 4.0:
        saran.append("⭐ **Tingkatkan kualitas** produk & layanan untuk mendongkrak rating ke atas 4.0.")
    if 0 < ulasan < 10:
        saran.append("💬 **Dorong pembeli** meninggalkan ulasan — tawarkan bonus kecil atau pesan follow-up.")
    if harga > STATS['pct75_harga']:
        saran.append(f"📈 Harga kamu masuk **top 25% termahal** (> Rp {STATS['pct75_harga']:,.0f}). Pastikan kualitas produk mendukung harga ini.")
    if prob >= 70:
        saran.append("🚀 Strategi sudah bagus! Fokus pada **konsistensi stok dan kecepatan pengiriman**.")

    if saran:
        for s in saran:
            st.markdown(f"- {s}")
    else:
        st.markdown("- Coba optimalkan kombinasi harga dan diskon untuk meningkatkan daya saing.")

# HEADER
st.title("🛍️ Prediksi Potensi Best Seller Produk")
st.write("Masukkan detail rencana produkmu di bawah ini, dan AI akan memprediksi seberapa besar peluang produk tersebut laku keras di pasaran.")

st.markdown("---")

tab_baru, tab_existing, tab_whatif, tab_pasar = st.tabs([
    "🆕 Produk Baru",
    "📦 Produk Existing",
    "🎛️ Simulasi What-If",        
    "📈 Analisis Pasar",          
])

with tab_baru:
    st.subheader("Simulasi Produk Baru")
    st.caption("Untuk produk yang belum punya rating & ulasan. Rating & ulasan diasumsikan 0.")

    col1, col2 = st.columns(2)
    with col1:
        harga_baru  = st.number_input("Harga Produk (Rp)", min_value=0, value=99000, step=5000, key="harga_baru")
        diskon_baru = st.slider("Rencana Diskon (%)", 0, 90, step=5, key="diskon_baru")
    with col2:
        harga_efektif_baru = harga_baru * (1 - diskon_baru / 100)
        st.metric("Harga Setelah Diskon", f"Rp {harga_efektif_baru:,.0f}")
        st.caption("Rating & ulasan = 0 karena produk baru.")

    if harga_baru == 0:
        st.warning("⚠️ Harga 0 tidak masuk akal. Masukkan harga yang valid.")

    if st.button("🔮 Prediksi", use_container_width=True, key="btn_baru"):
        kelas, prob = prediksi(harga_baru, diskon_baru, 0.0, 0)

        col_gauge, col_info = st.columns([1, 2])
        with col_gauge:
            # [BARU] Gauge chart
            st.markdown(gauge_chart(prob), unsafe_allow_html=True)
        with col_info:
            if kelas == 1:
                st.success("🔥 **BERPOTENSI BEST SELLER!**")
                st.balloons()
            else:
                st.warning("⚠️ **KURANG BERPOTENSI BEST SELLER**")
            st.markdown("**💡 Saran:**")
            tampilkan_saran(kelas, prob, harga_baru, diskon_baru, 0.0, 0)

        # [BARU] Benchmark vs pasar
        tampilkan_benchmark(harga_baru, diskon_baru, 0.0, 0)

with tab_existing:
    st.subheader("Analisis Produk Existing")
    st.caption("Untuk produk yang sudah berjalan dan punya data penjualan nyata.")

    col1, col2 = st.columns(2)
    with col1:
        harga_ex  = st.number_input("Harga Produk (Rp)", min_value=0, value=99000, step=5000, key="harga_ex")
        diskon_ex = st.slider("Diskon (%)", 0, 90, step=5, key="diskon_ex")
    with col2:
        rating_ex = st.slider("Rating Aktual (0.0–5.0)", 0.0, 5.0, value=4.5, step=0.1, key="rating_ex")
        ulasan_ex = st.number_input("Jumlah Ulasan Aktual", min_value=0, value=50, step=10, key="ulasan_ex")

    if st.button("🔮 Prediksi", use_container_width=True, key="btn_ex"):
        kelas, prob = prediksi(harga_ex, diskon_ex, rating_ex, ulasan_ex)

        col_gauge, col_info = st.columns([1, 2])
        with col_gauge:
            st.markdown(gauge_chart(prob), unsafe_allow_html=True)
        with col_info:
            if kelas == 1:
                st.success("🔥 **BERPOTENSI BEST SELLER!**")
                st.balloons()
            else:
                st.warning("⚠️ **KURANG BERPOTENSI BEST SELLER**")
            st.markdown("**💡 Saran:**")
            tampilkan_saran(kelas, prob, harga_ex, diskon_ex, rating_ex, ulasan_ex)

        tampilkan_benchmark(harga_ex, diskon_ex, rating_ex, ulasan_ex)

        # [BARU] Detail input
        with st.expander("🔍 Detail Input"):
            st.table(pd.DataFrame({
                "Parameter": ["Harga", "Diskon", "Harga Efektif", "Rating", "Ulasan", "Skor Kepercayaan"],
                "Nilai": [
                    f"Rp {harga_ex:,.0f}", f"{diskon_ex}%",
                    f"Rp {harga_ex*(1-diskon_ex/100):,.0f}",
                    f"{rating_ex}", f"{ulasan_ex}",
                    f"{rating_ex * ulasan_ex:.1f}",
                ]
            }))

with tab_whatif:
    st.subheader("🎛️ Simulasi What-If")
    st.caption(
        "Geser slider untuk eksplorasi secara real-time. "
        "Prediksi update otomatis tanpa perlu klik tombol."
    )

    col_slider, col_result = st.columns([1, 1])

    with col_slider:
        wi_harga  = st.slider("💰 Harga (Rp)", 5_000, 5_000_000, 99_000, step=5_000,
                              format="Rp %d", key="wi_harga")
        wi_diskon = st.slider("🏷️ Diskon (%)", 0, 90, 0, key="wi_diskon")
        wi_rating = st.slider("⭐ Rating", 0.0, 5.0, 4.5, step=0.1, key="wi_rating")
        wi_ulasan = st.slider("💬 Jumlah Ulasan", 0, 1000, 50, key="wi_ulasan")

        # [BARU] Tampilkan metrik turunan secara real-time
        wi_efektif = wi_harga * (1 - wi_diskon / 100)
        wi_skor    = wi_rating * wi_ulasan
        st.metric("Harga Efektif",    f"Rp {wi_efektif:,.0f}")
        st.metric("Skor Kepercayaan", f"{wi_skor:.1f}")

    with col_result:
        # [BARU] Prediksi real-time — tidak butuh tombol!
        # Streamlit re-run otomatis setiap slider bergerak.
        wi_kelas, wi_prob = prediksi(wi_harga, wi_diskon, wi_rating, wi_ulasan)

        st.markdown(gauge_chart(wi_prob), unsafe_allow_html=True)

        if wi_kelas == 1:
            st.success(f"🔥 **BERPOTENSI BEST SELLER** ({wi_prob:.1f}%)")
        else:
            st.warning(f"⚠️ **Kurang berpotensi** ({wi_prob:.1f}%)")

        # [BARU] Saran singkat real-time
        st.markdown("---")
        tampilkan_saran(wi_kelas, wi_prob, wi_harga, wi_diskon, wi_rating, wi_ulasan)

with tab_pasar:
    st.subheader("📈 Snapshot Pasar Tokopedia")
    st.caption(f"Berdasarkan {STATS['total_produk']:,} produk dalam dataset training.")

    # Statistik ringkas
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Produk Dianalisis", f"{STATS['total_produk']:,}")
    col2.metric("% Produk Best Seller",    f"{STATS['bestseller_rate']*100:.1f}%")
    col3.metric("Median Harga",            f"Rp {STATS['median_harga']:,.0f}")
    col4.metric("Threshold Best Seller",   f"{THRESHOLD:,.0f} pcs terjual")

    st.markdown("---")

    # Rentang harga pasar
    st.markdown("#### 💰 Distribusi Harga Pasar")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Harga Bawah (25%)", f"Rp {STATS['pct25_harga']:,.0f}", help="25% produk di bawah harga ini")
    col_b.metric("Median Harga",      f"Rp {STATS['median_harga']:,.0f}", help="Tepat di tengah pasar")
    col_c.metric("Harga Atas (75%)",  f"Rp {STATS['pct75_harga']:,.0f}", help="75% produk di bawah harga ini")

    st.markdown("#### 📦 Distribusi Penjualan")
    col_d, col_e, col_f = st.columns(3)
    col_d.metric("Terjual (Pct 25)",  f"{STATS['pct25_terjual']:,.0f} pcs")
    col_e.metric("Terjual (Pct 75)",  f"{STATS['pct75_terjual']:,.0f} pcs", help="Batas best seller")
    col_f.metric("Terjual (Top 10%)", f"{STATS['pct90_terjual']:,.0f} pcs", help="Top 10% produk terlaris")

    st.markdown("---")

    # Feature importance
    st.markdown("#### 🧠 Faktor Apa yang Paling Menentukan Best Seller?")
    st.caption("Semakin panjang bar, semakin besar pengaruh fitur tersebut terhadap prediksi model.")
    tampilkan_feature_importance()

    st.info(
        "💡 **Interpretasi**: Rating, Ulasan, dan Harga Efektif adalah tiga faktor teratas. "
        "Artinya: produk dengan rating tinggi, banyak ulasan, dan harga kompetitif memiliki "
        "peluang best seller yang jauh lebih tinggi dibanding produk yang hanya mengandalkan diskon."
    )

with st.sidebar:
    st.header("ℹ️ Tentang Model")
    st.markdown(f""")
**Algoritma**: Random Forest (200 pohon)

**Dilatih dari**: {STATS['total_produk']:,} produk Tokopedia

**Fitur yang digunakan** ({len(FEATURES)} total):
- Harga produk & harga setelah diskon
- Diskon (% dan ada/tidak)  
- Rating & jumlah ulasan
- Skor kepercayaan (rating × ulasan)

**Best Seller** = terjual > **{THRESHOLD:,.0f} pcs** (persentil 75)

**Performa model**: ROC-AUC ~0.89
(1.0 = sempurna, 0.5 = tebak acak)
    """)

    st.markdown("---")
    st.markdown("**🔍 Cara Baca Gauge:**")
    st.markdown("- 🔴 < 40% → Risiko tinggi tidak laku")
    st.markdown("- 🟡 40–65% → Cukup berpotensi")
    st.markdown("- 🟢 > 65% → Sangat berpotensi best seller")
    st.markdown("---")
    st.caption("Dibuat dengan ❤️ azek | Powered by scikit-learn & Streamlit")