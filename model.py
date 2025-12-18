import pandas as pd

# Load dataset
df = pd.read_csv("data/dtw_lpg_fixx.csv")

# Cek kategori unik
kategori_unik = df['kategori'].unique()

print("Kategori unik dalam dataset:")
for k in kategori_unik:
    print("-", k)

df.head(5)

print("Distribusi kategori:")
print(df['kategori'].value_counts())

fasilitas_set = set()
for row in df['Fasilitas']:
    items = [f.strip().title() for f in row.split(',')]
    fasilitas_set.update(items)

print("Fasilitas unik yang ditemukan:")
print(fasilitas_set)

df['kategori'] = df['kategori'].str.lower().str.strip()

# Mapping kategori
kategori_map = {
    'wisata alam': 'K1',
    'wisata buatan': 'K2',
    'wisata budaya': 'K3',
    'wisata religi': 'K4',
    'wisata lainnya': 'K5'
}

df['kategori_kode'] = df['kategori'].map(kategori_map)

# One-hot encoding kategori
for kode in ['K1', 'K2', 'K3', 'K4', 'K5']:
    df[kode] = (df['kategori_kode'] == kode).astype(int)


# Normalisasi fasilitas

def clean_fasilitas(fasilitas_str):
    if pd.isna(fasilitas_str):
        return []
    return [f.strip().title() for f in fasilitas_str.split(',')]

df['fasilitas_list'] = df['Fasilitas'].apply(clean_fasilitas)

# Mapping fasilitas dataset ke F1–F4
fasilitas_map = {
    'Parkir': 'F1',
    'Restoran': 'F2',
    'Toilet': 'F3',
    'Mushola': 'F4'
}

# One-hot encoding fasilitas (F1–F4)
for kode in ['F1', 'F2', 'F3', 'F4']:
    df[kode] = df['fasilitas_list'].apply(
        lambda x: 1 if fasilitas_map.get(x[0], None) == kode or kode in [fasilitas_map.get(item) for item in x] else 0
    )


# Tampilkan hasil

print("One-hot encoding kategori:")
print(df[['Nama tempat', 'K1','K2','K3','K4', 'K5']].head(10))

print("\nOne-hot encoding fasilitas:")
print(df[['Nama tempat', 'F1','F2','F3','F4']].head(10))

"""## **CBF**"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

#
# Fungsi deteksi kolom J dan F
def detect_k_f_cols(df, k_prefixes=('K',), f_prefixes=('F',)):
    k_cols = [c for c in df.columns if any(c.startswith(pref) for pref in k_prefixes) and c[1:].isdigit()]
    f_cols = [c for c in df.columns if any(c.startswith(pref) for pref in f_prefixes) and c[1:].isdigit()]
    k_cols = sorted(k_cols, key=lambda s: int(''.join(filter(str.isdigit, s))))
    f_cols = sorted(f_cols, key=lambda s: int(''.join(filter(str.isdigit, s))))
    return k_cols, f_cols

# Ambil matriks fitur CBF
def get_cbf_matrices(df):
    k_cols, f_cols = detect_k_f_cols(df, k_prefixes=('K',), f_prefixes=('F',))
    if len(k_cols) == 0 and len(f_cols) == 0:
        raise ValueError("Tidak ditemukan kolom K* atau F*. Pastikan one-hot sudah dibuat.")
    mat_k = df[k_cols].fillna(0).astype(float).values if len(k_cols) > 0 else None
    mat_f = df[f_cols].fillna(0).astype(float).values if len(f_cols) > 0 else None
    return mat_k, mat_f, k_cols, f_cols

# Hitung similarity (cosine)
def compute_similarity_matrices(df):
    mat_k, mat_f, k_cols, f_cols = get_cbf_matrices(df)
    sim_k = cosine_similarity(mat_k) if mat_k is not None else None
    sim_f = cosine_similarity(mat_f) if mat_f is not None else None
    return sim_k, sim_f, k_cols, f_cols

# Fungsi rekomendasi CBF
def recommend_cbf(df, target, top_n=10, alpha_k=0.5, alpha_f=0.5, by_name=False):
    """
    df        : dataframe yang sudah memiliki kolom J1.. Jk dan F1.. Fm
    target    : index (int) atau nama tempat (str) jika by_name=True
    top_n     : jumlah rekomendasi yang diinginkan
    alpha_j   : bobot untuk similarity kategori (J)
    alpha_f   : bobot untuk similarity fasilitas (F)
                NOTE: alpha_j + alpha_f should be 1.0 (but not strictly required)
    by_name   : jika True, 'target' dianggap sebagai string nama tempat di kolom 'Nama tempat'
    """
    sim_k, sim_f, k_cols, f_cols = compute_similarity_matrices(df)

    if by_name:
        if 'Nama tempat' not in df.columns:
            raise ValueError("Kolom 'Nama tempat' tidak ditemukan di DataFrame.")
        matches = df.index[df['Nama tempat'].str.lower() == str(target).lower()].tolist()
        if not matches:
            raise ValueError(f"Tidak ditemukan tempat dengan nama '{target}'. Pastikan penulisan benar.")
        target_idx = matches[0]
    else:
        target_idx = int(target)
        if target_idx not in df.index:
            if target_idx < 0 or target_idx >= len(df):
                raise IndexError("target index di luar jangkauan.")
            target_idx = df.index[target_idx]

    n = len(df)

    score_k = np.zeros(n)
    score_f = np.zeros(n)

    if sim_k is not None:
        score_k = sim_k[target_idx]
    if sim_f is not None:
        score_f = sim_f[target_idx]

    def minmax(arr):
        lo, hi = arr.min(), arr.max()
        if hi - lo <= 1e-9:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    score_k_n = minmax(score_k) if sim_k is not None else np.zeros_like(score_k)
    score_f_n = minmax(score_f) if sim_f is not None else np.zeros_like(score_f)

    final_score = alpha_k * score_k_n + alpha_f * score_f_n

    res = df.copy()
    res['cbf_score'] = final_score

    res = res.drop(index=target_idx)

    res = res.sort_values(by='cbf_score', ascending=False)

    return res.head(top_n).reset_index(drop=False).rename(columns={'index':'original_index'})

# Contoh pemakaian:
# contoh 1: pakai index posisi (misal 10)
#top5 = recommend_cbf(df, target=10, top_n=5, alpha_k=0.6, alpha_f=0.4, by_name=False)

# contoh 2: pakai nama tempat (case-insensitive)
top5 = recommend_cbf(df, target='Air Terjun Cengkaan', top_n=5, alpha_k=0.6, alpha_f=0.4, by_name=True)

print(top5[['Nama tempat', 'kabupaten kota', 'cbf_score']])

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def cbf_similarity(df):
    k_cols, f_cols = detect_k_f_cols(df)

    mat_k = df[k_cols].values
    mat_f = df[f_cols].values

    sim_k = cosine_similarity(mat_k)
    sim_f = cosine_similarity(mat_f)

    return sim_k, sim_f

def evaluate_similarity(df, sim_k, sim_f):
    results = {}
    kategori_unique = df['kategori'].unique()
    intra_list = []
    inter_list = []

    for kat in kategori_unique:
        idx_same = df.index[df['kategori'] == kat].tolist()
        idx_other = df.index[df['kategori'] != kat].tolist()

        if len(idx_same) > 1:
            intra_scores = sim_k[np.ix_(idx_same, idx_same)]
            upper_triangle = intra_scores[np.triu_indices(len(idx_same), k=1)]
            intra_list.extend(upper_triangle)

        if len(idx_same) > 0 and len(idx_other) > 0:
            inter_scores = sim_k[np.ix_(idx_same, idx_other)]
            inter_list.extend(inter_scores.flatten())

    results['intra_mean'] = np.mean(intra_list) if len(intra_list) else 0
    results['inter_mean'] = np.mean(inter_list) if len(inter_list) else 0
    results['separation'] = results['intra_mean'] - results['inter_mean']

    return results


# COVERAGE
def evaluate_coverage(sim_matrix, threshold=0.1):
    n = len(sim_matrix)
    count = 0
    for i in range(n):
        if np.sum(sim_matrix[i] > threshold) > 1:
            count += 1
    return count / n


# DIVERSITY
def evaluate_diversity(sim_matrix, top_n=5):
    n = len(sim_matrix)
    div_list = []

    for i in range(n):
        sims = sim_matrix[i].copy()
        sims[i] = -1
        top_idx = np.argsort(-sims)[:top_n]

        sub_sim = sim_matrix[np.ix_(top_idx, top_idx)]
        upper = sub_sim[np.triu_indices(top_n, k=1)]
        if len(upper) > 0:
            diversity = 1 - np.mean(upper)
            div_list.append(diversity)

    return np.mean(div_list) if div_list else 0


# EVALUASI CBF
def evaluate_cbf(df):
    sim_k, sim_f = cbf_similarity(df)

    print("\nEvaluasi Similarity (Kategori)")
    sim_eval = evaluate_similarity(df, sim_k, sim_f)
    print(sim_eval)

    print("\nCoverage")
    coverage = evaluate_coverage(sim_k)
    print(f"Coverage: {coverage:.3f}")

    print("\nDiversity")
    diversity = evaluate_diversity(sim_k, top_n=5)
    print(f"Diversity: {diversity:.3f}")

    return {
        "similarity_eval": sim_eval,
        "coverage": coverage,
        "diversity": diversity
    }

result = evaluate_cbf(df)
print(result)

"""## Hiversin"""

# Konversi kolom 'latitude' dan 'longitude' ke numerik
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

# Hapus baris yang memiliki NaN setelah konversi (jika ada nilai non-numerik asli)
df.dropna(subset=['latitude', 'longitude'], inplace=True)

# Reset index setelah menghapus baris untuk memastikan index berurutan
df.reset_index(drop=True, inplace=True)

# 2. Bersihkan coordinate outliers
df = df[(df['latitude'].between(-90,90)) & (df['longitude'].between(-180,180))]

# 3. Ubah rating menjadi float
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

df.head()

import pandas as pd
import numpy as np

# ==========================================================
# 1. LOAD DAN CLEANING DATA KOORDINAT
# ==========================================================

# Hapus duplikat
# Konversi latitude dan longitude ke numerik
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

# Hapus baris yang tidak memiliki koordinat
df = df.dropna(subset=['latitude', 'longitude'])

# Hapus koordinat outlier
df = df[
    df['latitude'].between(-90, 90) &
    df['longitude'].between(-180, 180)
]

# Reset index
df = df.reset_index(drop=True)


# ==========================================================
# 2. HAVERSINE VERSI OPTIMAL (VECTORIZED)
# ==========================================================

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Menghitung jarak Haversine secara vectorized.
    Bisa menerima scalar atau array NumPy.
    """
    R = 6371.0  # radius bumi (km)

    # Konversi dalam radian
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Selisih koordinat
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Rumus haversine
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


# ==========================================================
# 3. MENGHITUNG JARAK DARI TITIK REFERENSI
# ==========================================================

# Titik referensi (contoh: Bandarlampung)
ref_lat = -5.4295
ref_lon = 105.2620

df['jarak_km'] = haversine_vectorized(
    ref_lat,
    ref_lon,
    df['latitude'].values,
    df['longitude'].values
)

print(df[['Nama tempat', 'jarak_km']].head())


# ==========================================================
# 4. MATRIX JARAK ANTAR-DESTINASI (PAIRWISE)
# ==========================================================

def haversine_matrix(df):
    """
    Membuat matrix jarak NxN antar seluruh destinasi.
    Menggunakan vectorization penuh (tanpa loop, tanpa apply).
    """
    R = 6371.0

    lat = np.radians(df['latitude'].values)
    lon = np.radians(df['longitude'].values)

    # Membuat grid lat-lon
    lat1 = lat[:, None]
    lon1 = lon[:, None]
    lat2 = lat[None, :]
    lon2 = lon[None, :]

    # Selisih
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Rumus Haversine matrix
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    dist_matrix = 2 * R * np.arcsin(np.sqrt(a))

    return dist_matrix


# Buat matrix jarak
dist_matrix = haversine_matrix(df)

# Convert menjadi DataFrame
dist_df = pd.DataFrame(
    dist_matrix,
    index=df['Nama tempat'],
    columns=df['Nama tempat']
)

print(dist_df.head())

"""## LBS"""

import numpy as np
import pandas as pd

# ==============================================================
# 1. DATA CLEANING KOORDINAT
# ==============================================================

def clean_coordinates(df):
    df = df.copy()

    # Hapus duplikat
    df = df.drop_duplicates()

    # Konversi latitude dan longitude
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    # Hapus baris yang tidak punya koordinat
    df = df.dropna(subset=['latitude', 'longitude'])

    # Hapus koordinat outlier
    df = df[
        df['latitude'].between(-90, 90) &
        df['longitude'].between(-180, 180)
    ]

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ==============================================================
# 2. HAVERSINE VERSI OPTIMAL (VECTORIZED)
# ==============================================================

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Menghitung jarak menggunakan rumus Haversine secara vectorized,
    dapat menerima scalar atau array.
    """
    R = 6371.0  # radius bumi (km)

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


# ==============================================================
# 3. MATRIX JARAK ANTAR WISATA (PAIRWISE HAVERSINE)
# ==============================================================

def haversine_matrix(df):
    """
    Membuat matrix NxN berisi jarak antar item pada DataFrame.
    Tanpa loop, tanpa apply, full vectorized (sangat cepat).
    """
    R = 6371.0

    lat = np.radians(df['latitude'].values)
    lon = np.radians(df['longitude'].values)

    lat1 = lat[:, None]
    lon1 = lon[:, None]
    lat2 = lat[None, :]
    lon2 = lon[None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    dist_matrix = 2 * R * np.arcsin(np.sqrt(a))

    return dist_matrix


# ==============================================================
# 4. KONVERSI JARAK → SIMILARITY LBS
# ==============================================================

def lbs_similarity(df):
    """
    similarity = 1 / (1 + distance)
    Nilai mendekati 1 jika jarak sangat dekat.
    """
    dist_matrix = haversine_matrix(df)
    sim_lbs = 1 / (1 + dist_matrix)

    # Similarity diri sendiri = 0
    np.fill_diagonal(sim_lbs, 0)

    return sim_lbs


# ==============================================================
# 5. FUNGSI REKOMENDASI LBS
# ==============================================================

def recommend_lbs(df, target, top_n=10, by_name=True):
    """
    Mengembalikan rekomendasi wisata berdasarkan kedekatan lokasi.
    """
    sim_lbs = lbs_similarity(df)

    # Tentukan index target
    if by_name:
        idx = df.index[df['Nama tempat'].str.lower() == target.lower()]
        if len(idx) == 0:
            raise ValueError(f"Nama tempat '{target}' tidak ditemukan.")
        idx = idx[0]
    else:
        idx = target

    sims = sim_lbs[idx]

    # Ambil top-N similarity terbesar
    top_idx = np.argsort(-sims)[:top_n]

    hasil = df.iloc[top_idx].copy()
    hasil['lbs_score'] = sims[top_idx]

    return hasil[['Nama tempat', 'kabupaten kota', 'latitude', 'longitude', 'lbs_score']]


# ==============================================================
# 6. CONTOH PENGGUNAAN
# ==============================================================

rekom = recommend_lbs(df, "Curug Gimo", top_n=5)
print(rekom)

"""## Hybrid"""

def fix_unhashable(df):
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: str(x) if isinstance(x, (list, dict, set)) else x
        )
    return df

# 4) NORMALISASI (min-max)
# -------------------------
def minmax_normalize(arr):
    """
    Normalize array-like to 0..1. Works on 1D or 2D.
    If arr contains NaN (for matrix), ignore NaN when computing min/max but keep NaN positions.
    """
    a = np.array(arr, dtype=float)
    if a.ndim == 1:
        if np.all(np.isclose(a, a[0])):
            return np.zeros_like(a)
        lo = np.nanmin(a)
        hi = np.nanmax(a)
        if hi - lo <= 1e-12:
            return np.zeros_like(a)
        return (a - lo) / (hi - lo)
    else:
        # 2D matrix: normalize per entire matrix (flatten)
        flat = a.flatten()
        valid = ~np.isnan(flat)
        if valid.sum() == 0:
            return np.zeros_like(a)
        lo = np.nanmin(flat)
        hi = np.nanmax(flat)
        if np.isclose(hi, lo):
            return np.zeros_like(a)
        norm_flat = np.full_like(flat, np.nan, dtype=float)
        norm_flat[valid] = (flat[valid] - lo) / (hi - lo)
        return norm_flat.reshape(a.shape)

# -------------------------
# 5) BUILD HYBRID SIMILARITY
# -------------------------
def build_hybrid_similarity(df, alpha1=0.5, alpha2=0.3, beta=0.2):
    """
    Build hybrid similarity matrix:
    final = alpha1 * sim_j_norm + alpha2 * sim_f_norm + beta * sim_lbs_norm
    Requires alpha1+alpha2+beta ~= 1 (not strictly enforced but recommended).
    Handles missing coords: sim_lbs has zeros where coords missing.
    Returns hybrid_sim (NxN) matrix.
    """
    # 1. compute components
    sim_k, sim_f, _, _ = compute_similarity_matrices(df)
    sim_lbs = lbs_similarity(df)

    # If any sim_j or sim_f is None (e.g., no J columns), create zeros
    n = len(df)
    if sim_k is None:
        sim_k = np.zeros((n, n), dtype=float)
    if sim_f is None:
        sim_f = np.zeros((n, n), dtype=float)

    # 2. normalize each component to 0..1 (handle NaN in sim_lbs)
    sim_k_n = minmax_normalize(sim_k)
    sim_f_n = minmax_normalize(sim_f)
    sim_lbs_n = minmax_normalize(sim_lbs)

    # 3. combine
    hybrid = (alpha1 * sim_k_n) + (alpha2 * sim_f_n) + (beta * sim_lbs_n)

    # ensure diagonal 0 (no self-recommendation)
    np.fill_diagonal(hybrid, 0.0)

    return hybrid, sim_k_n, sim_f_n, sim_lbs_n

# -------------------------
# 6) REKOMENDASI HYBRID
# -------------------------
def recommend_hybrid(df, target, top_n=10, by_name=True, alpha1=0.5, alpha2=0.3, beta=0.2):
    """
    Return top_n hybrid recommendations for target (index or name).
    - alpha1, alpha2, beta should sum close to 1.
    - by_name=True uses 'Nama tempat' (case-insensitive).
    Output: DataFrame of recommendations containing component scores and final score.
    """
    hybrid, sim_k_n, sim_f_n, sim_lbs_n = build_hybrid_similarity(df, alpha1=alpha1, alpha2=alpha2, beta=beta)

    # resolve target index
    if by_name:
        if 'Nama tempat' not in df.columns:
            raise ValueError("Kolom 'Nama tempat' tidak ditemukan.")
        idx_list = df.index[df['Nama tempat'].str.lower() == str(target).lower()].tolist()
        if not idx_list:
            raise ValueError(f"Nama tempat '{target}' tidak ditemukan.")
        t = idx_list[0]
    else:
        t = int(target)
        if t < 0 or t >= len(df):
            raise IndexError("Index target di luar jangkauan.")

    scores = hybrid[t]
    # get top indices
    top_idx = np.argsort(-scores)[:top_n]
    res = df.iloc[top_idx].copy().reset_index(drop=True)

    # attach component scores
    res['score_hybrid'] = scores[top_idx]
    res['score_cat'] = sim_k_n[t, top_idx]
    res['score_fas'] = sim_f_n[t, top_idx]
    res['score_loc'] = sim_lbs_n[t, top_idx]

    # compute distance_km if lat/lon present (for display)
    if ('latitude' in df.columns) and ('longitude' in df.columns):
        # compute distances for target->top_idx using haversine_matrix (or vectorized haversine)
        # simple vectorized: haversine_vectorized(target_lat, target_lon, array_lats, array_lons)
        tlat = df.loc[t, 'latitude']
        tlon = df.loc[t, 'longitude']
        # if target has NaN coords, distances become NaN
        if not (np.isnan(tlat) or np.isnan(tlon)):
            # vectorized haversine for arrays
            lat_arr = res['latitude'].to_numpy(dtype=float)
            lon_arr = res['longitude'].to_numpy(dtype=float)
            # reuse haversine_vectorized logic:
            lat1 = np.radians(tlat)
            lon1 = np.radians(tlon)
            lat2 = np.radians(lat_arr)
            lon2 = np.radians(lon_arr)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
            dist_km = 2 * 6371.0 * np.arcsin(np.sqrt(a))
            res['distance_km'] = dist_km
        else:
            res['distance_km'] = np.nan

    return res

# -------------------------
# 7) EVALUASI HYBRID (simple)
# -------------------------
def evaluate_hybrid(df, alpha1=0.5, alpha2=0.3, beta=0.2):
    """
    Produce basic hybrid evaluation metrics similar to CBF:
    - intra_mean / inter_mean on combined sim (using category groups)
    - coverage (fraction of items with >=1 candidate > threshold)
    - diversity (1 - avg similarity among top-K recommendations)
    """
    hybrid, _, _, _ = build_hybrid_similarity(df, alpha1=alpha1, alpha2=alpha2, beta=beta)

    # similarity by categories
    cats = df['kategori'].unique()
    intra = []
    inter = []
    for cat in cats:
        idx_same = df.index[df['kategori'] == cat].tolist()
        idx_other = df.index[df['kategori'] != cat].tolist()
        if len(idx_same) > 1:
            mat = hybrid[np.ix_(idx_same, idx_same)]
            intra.extend(mat[np.triu_indices(len(idx_same), k=1)].tolist())
        if len(idx_same) > 0 and len(idx_other) > 0:
            mat2 = hybrid[np.ix_(idx_same, idx_other)]
            inter.extend(mat2.flatten().tolist())
    intra_mean = np.mean(intra) if len(intra)>0 else 0.0
    inter_mean = np.mean(inter) if len(inter)>0 else 0.0

    # coverage: at least one recommendation above small threshold
    n = len(hybrid)
    thresh = 0.01
    count = 0
    for i in range(n):
        if np.sum(hybrid[i] > thresh) >= 1:
            count += 1
    coverage = count / n

    # diversity (top-5)
    divs = []
    top_k = 5
    for i in range(n):
        sims = hybrid[i].copy()
        sims[i] = -1
        top_idx = np.argsort(-sims)[:top_k]
        sub = hybrid[np.ix_(top_idx, top_idx)]
        upper = sub[np.triu_indices(len(top_idx), k=1)]
        if len(upper) > 0:
            divs.append(1 - np.mean(upper))
    diversity = np.mean(divs) if len(divs)>0 else 0.0

    return {
        "intra_mean": intra_mean,
        "inter_mean": inter_mean,
        "separation": intra_mean - inter_mean,
        "coverage": coverage,
        "diversity": diversity
    }

"""## Penggunaan Hybrid Recommendation CBF + LBS"""

df_fixed = fix_unhashable(df)
df_clean = clean_coordinates(df_fixed)

hybrid_sim, _, _, _ = build_hybrid_similarity(df_clean, alpha1=0.5, alpha2=0.3, beta=0.2)
rec = recommend_hybrid(df_clean, target="Air Terjun Cengkaan", top_n=10, by_name=True, alpha1=0.5, alpha2=0.3, beta=0.2)
print(rec[['Nama tempat','kategori','score_hybrid','score_cat','score_fas','score_loc','distance_km']])
eval_result = evaluate_hybrid(df_clean, alpha1=0.5, alpha2=0.3, beta=0.2)
print(eval_result)