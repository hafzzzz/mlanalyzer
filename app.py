"""
ML Studio - titik masuk aplikasi.   Jalankan:  streamlit run app.py

Struktur proyek:
    app.py              orkestrasi UI (file ini)
    mlstudio/core.py    logika data & ML (tanpa Streamlit/Plotly, teruji)
    mlstudio/charts.py  pembuat grafik Plotly
    mlstudio/ui.py      komponen tampilan (CSS, kartu, KPI)
    tests/              pytest untuk core
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from mlstudio import charts, core, ui

st.set_page_config(page_title="ML Studio", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
ui.inject_css()
charts.setup_theme()
stepper_slot = st.empty()


@st.cache_data(show_spinner="Membaca dataset...")
def cached_read(content: bytes, name: str) -> pd.DataFrame:
    return core.read_table(content, name)


@st.cache_data(show_spinner=False)
def cached_sample(key: str) -> pd.DataFrame:
    return core.load_sample(key)


SAMPLE_META = {"wine": ("flask", "violet"), "diabetes": ("activity", "cyan")}


# ============================================================================ sidebar 1
with st.sidebar:
    ui.H(f'<div class="side-brand">{ui.logo(True)}<div><b>ML Studio</b><span>Data analyzer</span></div></div>')
    ui.H('<div class="side-label">1 &middot; Dataset</div>')
    uploaded = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls"],
                                help="Format yang didukung: CSV, XLSX, XLS.", label_visibility="collapsed")
    if uploaded is None and st.session_state.get("sample"):
        if st.button("Kembali ke halaman awal", key="reset_sample", width="stretch", icon=":material/arrow_back:"):
            st.session_state["sample"] = None
            st.session_state.pop("exp", None)
            st.rerun()


def landing() -> None:
    stepper_slot.markdown(ui.stepper_html(0), unsafe_allow_html=True)
    ui.H(f"""<div class="hero"><div><div class="eyebrow">ML Studio</div>
        <h1>Dari dataset mentah ke model terbaik, dengan penjelasan di setiap langkah.</h1>
        <p>Unggah CSV atau Excel. ML Studio memeriksa kualitas data, membandingkan model dengan baseline dan cross-validation,
        lalu menulis catatan analis yang menjelaskan hasilnya.</p>
        {ui.tags(["Data Quality", "Baseline", "Cross-validation", "Catatan analis", "Skrip reproduksi"], "x")}
        </div><div class="hero-art">{ui.HERO_ART}</div></div>""")
    feats = [
        ("shield", "green", "Audit kualitas data", "Missing value, duplikat, kolom ID/konstan, korelasi tinggi, hingga dugaan kebocoran data."),
        ("zap", "amber", "Evaluasi yang jujur", "Setiap model dibandingkan dengan baseline naif, dinilai lewat cross-validation, dan dicek overfitting."),
        ("code", "brand", "Transparan & bisa diaudit", "Dapatkan laporan Markdown dan skrip Python mandiri untuk mereproduksi model di luar aplikasi."),
    ]
    for c, (ic, tone, title, text) in zip(st.columns(3, gap="medium"), feats):
        bg, fg = ui.TONES[tone]
        c.markdown(ui.compress(f'<div class="feat"><div class="ico" style="background:{bg};color:{fg}">{ui.icon(ic, 22)}</div>'
                               f'<b>{title}</b><p>{text}</p></div>'), unsafe_allow_html=True)
    ui.section("Mulai cepat", "Belum punya data? Coba dataset contoh", "Klik salah satu untuk langsung menjelajahi seluruh fitur.")
    for c, (key, meta) in zip(st.columns(2, gap="medium"), core.SAMPLE_DATASETS.items()):
        with c, ui.card():
            ic, tone = SAMPLE_META[key]
            ui.icon_row(ic, tone, meta["label"], f'{meta["kind"]} · {meta["desc"]}')
            if st.button("Gunakan dataset ini", key=f"sample_{key}", width="stretch", icon=":material/play_arrow:"):
                st.session_state["sample"] = key
                st.session_state.pop("exp", None)
                st.rerun()
    ui.footnote("Unggah file dari panel kiri untuk menganalisis data Anda sendiri. Format: CSV, XLSX, XLS.")


# ============================================================================ data aktif
if uploaded is not None:
    try:
        df = cached_read(uploaded.getvalue(), uploaded.name)
    except core.DataError as exc:
        stepper_slot.markdown(ui.stepper_html(0), unsafe_allow_html=True)
        ui.callout("bad", "Gagal membaca file", str(exc), "Pastikan file CSV/Excel valid dan tidak rusak.")
        st.stop()
    source_name = uploaded.name
elif st.session_state.get("sample"):
    df = cached_sample(st.session_state["sample"])
    source_name = core.SAMPLE_DATASETS[st.session_state["sample"]]["label"] + " (contoh)"
else:
    landing()
    st.stop()

if df.empty:
    stepper_slot.markdown(ui.stepper_html(0), unsafe_allow_html=True)
    ui.callout("bad", "Dataset kosong", "File berhasil dibaca tetapi tidak berisi data.")
    st.stop()

profile = core.profile_dataset(df)
quality = core.quality_findings(df, profile)

# ============================================================================ sidebar 2
with st.sidebar:
    ui.H('<div class="side-label">2 &middot; Target &amp; model</div>')
    target = st.selectbox("Kolom target", options=list(df.columns), index=len(df.columns) - 1,
                          help="Kolom yang ingin diprediksi.")
    problem_choice = st.selectbox("Jenis masalah", [core.AUTO_DETECT, core.CLASSIFICATION, core.REGRESSION],
                                  help="Auto-detect: klasifikasi bila target non-numerik atau berisi 15 nilai unik atau kurang.")
    ui.H('<div class="side-label">3 &middot; Evaluasi</div>')
    test_size = st.slider("Porsi data uji (%)", 10, 40, 20, 5) / 100
    cv_folds = st.select_slider("Cross-validation (fold)", options=[0, 3, 5, 10], value=5,
                                help="Validasi silang pada data latih untuk skor yang lebih stabil. 0 = nonaktif (lebih cepat).")
    seed = int(st.number_input("Random state", 0, 9999, 42, help="Angka acak tetap agar hasil dapat diulang."))
    refit_full = st.toggle("Latih ulang model final di seluruh data", value=True,
                           help="Evaluasi tetap memakai data uji; model yang diunduh dilatih ulang di seluruh data agar memakai semua informasi.")
    ui.H(f'<div class="side-card">Sumber data<br><b>{ui.esc(source_name)}</b><br>{len(df):,} baris &middot; {df.shape[1]} kolom</div>')
    ui.H('<div class="side-foot">ML Studio v2 &middot; Streamlit, scikit-learn, Plotly.</div>')

ui.topbar(source_name, profile.n_rows, profile.n_cols)

tab_overview, tab_eda, tab_ml, tab_predict, tab_guide = st.tabs(
    ["Ringkasan", "Eksplorasi", "Modeling", "Prediksi", "Panduan"])


# ============================================================================ tab 1
with tab_overview:
    ui.section("01 · Data", "Ringkasan dataset", "Struktur, kualitas, dan profil tiap kolom dalam satu tampilan.")
    k = st.columns(4, gap="medium")
    ui.kpi(k[0], "Baris", f"{profile.n_rows:,}", "observasi", "database", "brand")
    ui.kpi(k[1], "Kolom", f"{profile.n_cols:,}", f"{len(profile.numeric)} numerik · {len(profile.categorical)} kategorikal", "columns", "cyan")
    ui.kpi(k[2], "Baris duplikat", f"{profile.duplicate_rows:,}", f"{100 * (1 - profile.uniqueness):.1f}% dari total",
           "copy", "amber" if profile.duplicate_rows else "green")
    ui.kpi(k[3], "Sel kosong", f"{profile.missing_cells:,}", f"{100 * (1 - profile.completeness):.1f}% dari seluruh sel",
           "alert", "rose" if profile.missing_cells else "green")
    ui.gap(14)
    left, right = st.columns([1, 1.5], gap="medium")
    with left, ui.card():
        ui.card_title("Skor kualitas data", "Kelengkapan (70%) + keunikan baris (30%).")
        ui.quality_ring(profile.quality_score, profile.completeness, profile.uniqueness)
    with right, ui.card():
        ui.card_title("Temuan & saran", "Diperiksa otomatis dari isi dataset.")
        ui.findings(quality)

    ui.section("02 · Profil", "Profil kolom", "Tipe data, kelengkapan, dan kardinalitas tiap kolom.")
    prof_df = core.column_profile(df)
    with ui.card():
        st.dataframe(prof_df, width="stretch", hide_index=True, column_config={
            "Missing %": st.column_config.ProgressColumn("Missing %", min_value=0, max_value=100, format="%.1f%%")})
    if (prof_df["Missing"] > 0).any():
        ui.gap(14)
        with ui.card():
            ui.card_title("Nilai kosong per kolom")
            ui.show(charts.missing_bar(prof_df), height=max(220, 34 * int((prof_df["Missing"] > 0).sum()) + 60))
    ui.gap(14)
    with st.expander("Pratinjau dataset (100 baris pertama)"):
        st.dataframe(df.head(100), width="stretch")


# ============================================================================ tab 2
with tab_eda:
    ui.section("03 · Eksplorasi", "Eksplorasi data", "Pahami sebaran, hubungan, dan pola sebelum modeling.")
    t_stat, t_dist, t_corr, t_rel, t_tgt = st.tabs(["Statistik", "Distribusi", "Korelasi", "Hubungan", "Fitur vs target"])
    num_cols, cat_cols = list(profile.numeric), list(profile.categorical)

    with t_stat, ui.card():
        ui.card_title("Statistik deskriptif", "Kolom numerik.")
        if num_cols:
            st.dataframe(df[num_cols].describe().T.round(3), width="stretch")
        else:
            ui.callout("info", "Tidak ada kolom numerik", "Dataset ini hanya berisi kolom kategorikal.")
        if cat_cols:
            ui.card_title("Ringkasan kategorikal", "Nilai unik dan nilai yang paling sering muncul.")
            rows = []
            for c in cat_cols:
                vc = df[c].astype(str).value_counts()
                rows.append({"Kolom": c, "Unik": int(df[c].nunique()), "Nilai teratas": vc.index[0], "Frekuensi": int(vc.iloc[0])})
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with t_dist:
        d1, d2 = st.columns(2, gap="medium")
        with d1, ui.card():
            ui.card_title("Distribusi numerik", "Histogram + box plot, dengan catatan bentuk distribusi.")
            if num_cols:
                col = st.selectbox("Kolom numerik", num_cols, key="dist_num")
                ui.show(charts.histogram(df, col), height=360)
                ui.H("".join(f'<div class="c-sub">• {ui.esc(t)}</div>' for t in core.describe_distribution(df[col])))
            else:
                ui.callout("info", "Tidak tersedia", "Tidak ada kolom numerik.")
        with d2, ui.card():
            ui.card_title("Frekuensi kategorikal", "Maksimal 15 nilai teratas.")
            if cat_cols:
                col = st.selectbox("Kolom kategorikal", cat_cols, key="dist_cat")
                ui.show(charts.category_bar(df, col), height=360)
            else:
                ui.callout("info", "Tidak tersedia", "Tidak ada kolom kategorikal.")

    with t_corr:
        if len(num_cols) >= 2:
            c1, c2 = st.columns([1.8, 1], gap="medium")
            with c1, ui.card():
                ui.card_title("Matriks korelasi (Pearson)", "Biru: positif, merah: negatif.")
                ui.show(charts.corr_heatmap(df[num_cols].corr()), height=max(380, 34 * len(num_cols) + 120))
            with c2, ui.card():
                ui.card_title("Pasangan terkuat", "Berdasarkan |r|. Korelasi bukan sebab-akibat.")
                st.dataframe(core.top_correlations(df, num_cols).round(3), width="stretch", hide_index=True)
        else:
            with ui.card():
                ui.callout("info", "Korelasi tidak tersedia", "Dibutuhkan minimal dua kolom numerik.")

    with t_rel, ui.card():
        ui.card_title("Hubungan antar fitur", "Scatter dua kolom numerik, warna opsional.")
        if len(num_cols) >= 2:
            r1, r2, r3 = st.columns(3)
            x_col = r1.selectbox("Sumbu X", num_cols, index=0, key="rel_x")
            y_col = r2.selectbox("Sumbu Y", num_cols, index=1, key="rel_y")
            options = ["(tanpa warna)"] + [c for c in df.columns if df[c].nunique() <= 12]
            color = r3.selectbox("Warna berdasarkan", options, key="rel_c")
            ui.show(charts.scatter(df, x_col, y_col, None if color == "(tanpa warna)" else color), height=420)
        else:
            ui.callout("info", "Tidak tersedia", "Dibutuhkan minimal dua kolom numerik.")

    with t_tgt, ui.card():
        problem_for_eda = core.infer_problem(df[target].dropna(), problem_choice) if df[target].notna().any() else core.CLASSIFICATION
        ui.card_title(f"Keterkaitan fitur dengan target `{target}`",
                      "Skrining cepat 0-1 sebelum modeling. Metode dipilih otomatis menurut tipe data. Bukan bukti sebab-akibat.")
        assoc = core.feature_target_association(df, target, problem_for_eda)
        if assoc.empty:
            ui.callout("info", "Tidak dapat dihitung", "Target atau fitur tidak memiliki variasi/data yang cukup.")
        else:
            a1, a2 = st.columns([1.5, 1], gap="medium")
            with a1:
                ui.show(charts.association_bar(assoc), height=max(300, 28 * min(len(assoc), 15) + 80))
            with a2:
                st.dataframe(assoc.round(3), width="stretch", hide_index=True)


# ============================================================================ tab 3
def render_results(exp: dict) -> None:
    prep: core.PreparedData = exp["prep"]
    res: core.ExperimentResult = exp["res"]
    lb, metric = res.leaderboard, res.primary_metric
    best = lb[lb["Model"] == res.best_name].iloc[0]
    y_test, y_pred = res.split.y_test, res.y_pred
    is_clf = prep.problem == core.CLASSIFICATION

    ui.section("05 · Hasil", "Hasil & evaluasi", "Kesimpulan dalam bahasa sederhana, lalu detail angkanya.")
    pills = ([("Accuracy", best["Accuracy"]), ("Precision", best["Precision (macro)"]), ("Recall", best["Recall (macro)"])]
             if is_clf else [("MAE", best["MAE"]), ("RMSE", best["RMSE"])])
    ui.winner_banner(res.best_name, float(best[metric]), metric, pills)

    with ui.card():
        ui.card_title("Catatan analis", f"Dihasilkan otomatis dari hasil eksperimen ({res.seconds:.1f} detik).")
        ui.findings(exp["notes"])
    ui.gap(14)

    r_lb, r_eval, r_imp, r_code, r_dl = st.tabs(["Leaderboard", "Evaluasi", "Fitur penting", "Reproduksi", "Unduh"])

    with r_lb:
        with ui.card():
            ui.card_title("Leaderboard", f"Diurutkan berdasarkan {res.rank_by}"
                          + (f" ({res.cv_folds_used}-fold CV pada data latih)." if res.cv_folds_used else " (data uji)."))
            num_cfg = {c: st.column_config.NumberColumn(c, format="%.3f") for c in lb.columns
                       if c not in ("#", "Model") and pd.api.types.is_float_dtype(lb[c])}
            if is_clf:
                for c in ("Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"):
                    num_cfg[c] = st.column_config.ProgressColumn(c, min_value=0.0, max_value=1.0, format="%.3f")
            st.dataframe(lb, width="stretch", hide_index=True, column_config=num_cfg)
            st.caption("Δ vs baseline: selisih skor terhadap tebakan naif. Gap: skor latih dikurangi skor uji (besar = overfit).")
        ui.gap(14)
        c1, c2 = st.columns(2, gap="medium")
        with c1, ui.card():
            ui.card_title(f"Skor {res.rank_by}", "Biru: terbaik · Kuning: baseline.")
            ui.show(charts.leaderboard_bar(lb, res.rank_by, res.best_name, core.BASELINE_NAME), height=max(300, 44 * len(lb) + 70))
        with c2, ui.card():
            ui.card_title("Skor latih vs uji", "Jarak lebar = model menghafal data latih.")
            ui.show(charts.train_vs_test(lb, metric), height=max(300, 52 * len(lb) + 70))

    with r_eval:
        if is_clf:
            names = [str(c) for c in prep.label_encoder.classes_]
            m = st.columns(4, gap="medium")
            ui.kpi(m[0], "Accuracy", f"{best['Accuracy']:.3f}", "ketepatan keseluruhan", "target", "brand")
            ui.kpi(m[1], "Precision", f"{best['Precision (macro)']:.3f}", "rata-rata makro", "check", "green")
            ui.kpi(m[2], "Recall", f"{best['Recall (macro)']:.3f}", "rata-rata makro", "layers", "cyan")
            ui.kpi(m[3], "F1-score", f"{best['F1 (macro)']:.3f}", "rata-rata makro", "zap", "amber")
            ui.gap(14)
            labels = np.arange(len(names))
            left, right = st.columns([1.1, 1], gap="medium")
            with left, ui.card():
                ui.card_title("Confusion matrix", "Baris: kelas aktual · kolom: kelas prediksi.")
                norm = st.toggle("Tampilkan dalam persen (per kelas aktual)", value=False, key="cm_norm")
                ui.show(charts.confusion(confusion_matrix(y_test, y_pred, labels=labels), names, norm),
                        height=max(360, 56 * len(names) + 120))
            with right, ui.card():
                ui.card_title("Classification report", "Metrik per kelas pada data uji.")
                rep = classification_report(y_test, y_pred, labels=labels, target_names=names, output_dict=True, zero_division=0)
                st.dataframe(pd.DataFrame(rep).T.round(3), width="stretch")
        else:
            m = st.columns(3, gap="medium")
            ui.kpi(m[0], "MAE", f"{best['MAE']:.3f}", "rata-rata galat absolut", "target", "brand")
            ui.kpi(m[1], "RMSE", f"{best['RMSE']:.3f}", "menghukum galat besar", "activity", "amber")
            ui.kpi(m[2], "R²", f"{best['R2']:.3f}", "variasi yang dijelaskan", "zap", "green")
            ui.gap(14)
            c1, c2 = st.columns(2, gap="medium")
            with c1, ui.card():
                ui.card_title("Aktual vs prediksi", "Titik dekat garis putus-putus = prediksi akurat.")
                ui.show(charts.actual_vs_predicted(y_test, y_pred), height=380)
            with c2, ui.card():
                ui.card_title("Distribusi residual", "Idealnya simetris dan berpusat di nol.")
                ui.show(charts.residuals(y_test, y_pred), height=380)

    with r_imp, ui.card():
        ui.card_title("Fitur paling berpengaruh", "Permutation importance pada data uji: penurunan skor saat fitur diacak. Garis galat = simpangan baku antar pengacakan.")
        if res.importance is not None and not res.importance.empty:
            ui.show(charts.importance_bar(res.importance), height=max(320, 30 * len(res.importance) + 70))
        else:
            ui.callout("warn", "Feature importance tidak tersedia", res.importance_error or "Tidak dapat dihitung.")

    with r_code, ui.card():
        ui.card_title("Skrip reproduksi", "Python mandiri (pandas + scikit-learn). Ubah DATA_PATH ke file Anda, lalu jalankan.")
        st.code(exp["script"], language="python")
        st.download_button("Unduh skrip (.py)", data=exp["script"].encode("utf-8"), file_name="reproduce_model.py",
                           mime="text/x-python", icon=":material/download:")

    with r_dl:
        test_out = pd.DataFrame({"Actual": np.asarray(y_test), "Predicted": np.asarray(y_pred)})
        if is_clf:
            test_out["Actual"] = prep.label_encoder.inverse_transform(test_out["Actual"].astype(int))
            test_out["Predicted"] = prep.label_encoder.inverse_transform(test_out["Predicted"].astype(int))
        items = [
            ("Laporan analisis", "Ringkasan lengkap (Markdown) untuk dibagikan.", "file", "violet",
             exp["report"].encode("utf-8"), "analysis_report.md", "text/markdown"),
            ("Model terbaik", "Pipeline lengkap (.joblib) + metadata fitur.", "layers", "brand",
             exp["bundle"], "best_ml_model.joblib", "application/octet-stream"),
            ("Leaderboard", "Seluruh metrik tiap algoritma (CSV).", "bars", "cyan",
             lb.to_csv(index=False).encode("utf-8"), "model_comparison.csv", "text/csv"),
            ("Prediksi data uji", "Aktual vs prediksi (CSV).", "target", "green",
             test_out.to_csv(index=False).encode("utf-8"), "test_predictions.csv", "text/csv"),
        ]
        for c, (title, desc, ic, tone, data, fname, mime) in zip(st.columns(4, gap="medium"), items):
            with c, ui.card():
                ui.icon_row(ic, tone, title, desc)
                st.download_button("Unduh", data=data, file_name=fname, mime=mime, key=f"dl_{fname}",
                                   width="stretch", icon=":material/download:")
        if res.refit_full:
            st.caption("Model .joblib dilatih ulang di seluruh data (evaluasi di atas tetap memakai data uji). "
                       "Gunakan tab Prediksi untuk memakainya pada data baru.")


def render_modeling() -> None:
    ui.section("04 · Modeling", "Persiapan & pelatihan", "Cek konfigurasi, lalu latih dan bandingkan beberapa algoritma sekaligus.")
    try:
        prep = core.prepare_data(df, target, problem_choice)
        split = core.split_data(prep, test_size, seed)
    except core.DataError as exc:
        ui.callout("bad", "Konfigurasi belum valid", str(exc), "Ubah kolom target, jenis masalah, atau porsi data uji di sidebar.")
        return

    is_clf = prep.problem == core.CLASSIFICATION
    k = st.columns(4, gap="medium")
    ui.kpi(k[0], "Target", str(target), f"{df[target].nunique():,} nilai unik", "target", "rose")
    ui.kpi(k[1], "Jenis masalah", "Klasifikasi" if is_clf else "Regresi",
           "terdeteksi otomatis" if problem_choice == core.AUTO_DETECT else "dipilih manual", "layers", "brand")
    ui.kpi(k[2], "Latih / uji", f"{len(split.y_train):,} / {len(split.y_test):,}", f"seed {seed}", "split", "cyan")
    ui.kpi(k[3], "Fitur", f"{len(prep.feature_columns)}", f"{len(prep.numeric_features)} numerik · {len(prep.categorical_features)} kategorikal", "columns", "violet")
    ui.gap(14)

    left, right = st.columns(2, gap="medium")
    with left, ui.card():
        ui.card_title("Distribusi target", "Pastikan sebaran kelas/nilai masuk akal.")
        if is_clf:
            ui.show(charts.target_distribution(counts=prep.class_counts), height=300)
        else:
            ui.show(charts.target_distribution(values=pd.to_numeric(df[target], errors="coerce").dropna()), height=300)
    with right, ui.card():
        ui.card_title("Persiapan data otomatis", "Keputusan yang diambil sebelum pelatihan.")
        ui.H(ui.tags([f"{c}" for c in prep.numeric_features[:40]], "num") + ui.tags([f"{c}" for c in prep.categorical_features[:40]], "cat"))
        ui.gap(8)
        if prep.findings:
            ui.findings(prep.findings)
        else:
            ui.callout("good", "Tidak ada penyesuaian", "Semua kolom dipakai sebagai fitur.")
        ui.H('<div class="c-sub">Pipeline: numerik → imputasi median + standardisasi · kategorikal → imputasi modus + one-hot.</div>')

    zoo = core.model_zoo(prep.problem, seed, len(prep.y))
    signature = (source_name, len(df), str(target), prep.problem, test_size, cv_folds, seed, refit_full)
    ui.gap(14)
    with ui.card():
        c_text, c_btn = st.columns([2.4, 1], gap="medium", vertical_alignment="center")
        with c_text:
            ui.card_title(f"Siap melatih {len(zoo)} algoritma?",
                          f"{len(split.y_train):,} baris latih · {len(split.y_test):,} baris uji · "
                          + (f"{cv_folds}-fold cross-validation." if cv_folds else "tanpa cross-validation."))
            ui.H(ui.tags([m.name for m in zoo], "mod"))
        with c_btn:
            clicked = st.button("Latih & bandingkan", type="primary", key="train", width="stretch", icon=":material/rocket_launch:")
    with st.expander("Apa saja algoritma ini?"):
        st.dataframe(core.models_overview(prep.problem, seed, len(prep.y)), width="stretch", hide_index=True)

    if clicked:
        with st.status("Melatih model...", expanded=True) as status:
            bar, msg = st.progress(0.0), st.empty()

            def on_progress(i: int, n: int, name: str) -> None:
                bar.progress(min(i / max(n, 1), 1.0))
                msg.write(f"Melatih **{name}**" if i < n else "Menghitung feature importance & finalisasi...")

            try:
                res = core.run_experiment(prep, split, seed=seed, cv_folds=cv_folds, refit_full=refit_full, progress=on_progress)
            except core.DataError as exc:
                status.update(label="Pelatihan gagal", state="error")
                ui.callout("bad", "Pelatihan gagal", str(exc))
                return
            notes = core.analyst_notes(prep, res)
            st.session_state["exp"] = dict(
                signature=signature, prep=prep, res=res, notes=notes,
                report=core.build_report(source_name, profile, quality, prep, res, notes, test_size, seed),
                bundle=core.bundle_to_bytes(prep, res),
                script=core.reproducible_script(source_name, prep, res.best_name, test_size, seed),
            )
            status.update(label=f"Selesai dalam {res.seconds:.1f} detik", state="complete", expanded=False)

    exp = st.session_state.get("exp")
    if exp is None:
        ui.gap(10)
        ui.callout("info", "Belum ada hasil", "Klik Latih & bandingkan untuk memulai.")
    elif exp["signature"] != signature:
        ui.gap(10)
        ui.callout("warn", "Pengaturan berubah", "Dataset atau pengaturan berubah sejak pelatihan terakhir.", "Latih ulang untuk memperbarui hasil.")
    else:
        render_results(exp)


with tab_ml:
    render_modeling()


# ============================================================================ tab 4
with tab_predict:
    ui.section("06 · Prediksi", "Prediksi data baru", "Pakai model terbaik pada data baru yang punya kolom fitur yang sama.")
    exp = st.session_state.get("exp")
    if exp is None:
        with ui.card():
            ui.callout("info", "Latih model terlebih dahulu", "Buka tab Modeling lalu klik Latih & bandingkan.")
    else:
        prep, res = exp["prep"], exp["res"]
        if exp["signature"][2] != str(target):
            ui.callout("warn", "Model dari target lain", f"Model ini dilatih untuk target `{prep.target}`, bukan `{target}`.")
        with ui.card():
            ui.card_title(f"Model aktif: {res.best_name}", f"Target `{prep.target}` · {len(prep.feature_columns)} fitur wajib ada di file baru.")
            tpl = prep.X.head(3).to_csv(index=False).encode("utf-8")
            st.download_button("Unduh template CSV (3 baris contoh)", data=tpl, file_name="prediction_template.csv",
                               mime="text/csv", icon=":material/download:")
            new_file = st.file_uploader("Upload data baru (tanpa kolom target)", type=["csv", "xlsx", "xls"], key="new_data")
        if new_file is not None:
            try:
                new_df = core.read_table(new_file.getvalue(), new_file.name)
                preds = core.predict_new(prep, res, new_df)
            except core.DataError as exc:
                ui.callout("bad", "Prediksi tidak dapat dijalankan", str(exc), "Samakan nama kolom dengan template.")
            else:
                ui.gap(14)
                with ui.card():
                    ui.card_title("Hasil prediksi", f"{len(preds):,} baris.")
                    st.dataframe(preds, width="stretch")
                    st.download_button("Unduh hasil (.csv)", data=preds.to_csv(index=False).encode("utf-8"),
                                       file_name="predictions.csv", mime="text/csv", icon=":material/download:")
                if prep.problem == core.CLASSIFICATION:
                    ui.callout("info", "Cara membaca confidence", "Nilai ini adalah probabilitas kelas terpilih menurut model, bukan jaminan kebenaran.",
                               "Nilai rendah (< 0.6) layak ditinjau manual.")


# ============================================================================ tab 5
with tab_guide:
    ui.section("Panduan", "Cara kerja & cara membaca hasil", "Ringkasan alur, istilah, dan batasan, supaya angkanya tidak disalahartikan.")
    steps = [
        ("1 · Inspeksi", "Data diperiksa: missing value, duplikat, kolom konstan/ID, korelasi tinggi. Tidak ada data yang diubah di file Anda."),
        ("2 · Persiapan", "Kolom konstan & ID dibuang, target di-encode, dan dugaan kebocoran data dicek. Numerik: median + standardisasi. Kategorikal: modus + one-hot."),
        ("3 · Pembagian", "Data dibagi latih/uji (stratified untuk klasifikasi). Data uji disimpan untuk evaluasi akhir dan tidak dipakai saat memilih model dengan CV."),
        ("4 · Pelatihan", "Setiap algoritma dilatih dalam pipeline yang sama. Baseline naif ikut dilatih sebagai patokan minimum."),
        ("5 · Pemilihan", "Peringkat memakai skor cross-validation (lebih stabil dari satu split). Baseline tidak pernah dipilih sebagai terbaik."),
        ("6 · Interpretasi", "Catatan analis menilai: selisih dengan baseline, overfitting, kestabilan antar fold, ukuran data uji, kelas tersulit, dan ketergantungan pada satu fitur."),
    ]
    for c, group in zip(st.columns(2, gap="medium"), (steps[:3], steps[3:])):
        with c, ui.card():
            for title, text in group:
                ui.H(f'<div class="c-title">{ui.esc(title)}</div><div class="c-sub" style="margin-bottom:12px">{ui.esc(text)}</div>')
    ui.gap(14)
    with ui.card():
        ui.card_title("Glosarium metrik", "Arti istilah yang muncul di hasil, dan kapan perlu waspada.")
        st.dataframe(core.glossary_table(), width="stretch", hide_index=True)
    ui.gap(14)
    g1, g2 = st.columns(2, gap="medium")
    with g1, ui.card():
        ui.card_title("Algoritma klasifikasi")
        st.dataframe(core.models_overview(core.CLASSIFICATION), width="stretch", hide_index=True)
    with g2, ui.card():
        ui.card_title("Algoritma regresi")
        st.dataframe(core.models_overview(core.REGRESSION), width="stretch", hide_index=True)
    ui.gap(14)
    with ui.card():
        ui.card_title("Batasan yang perlu diingat")
        ui.findings([
            core.Finding("info", "Korelasi bukan sebab-akibat", "Fitur yang berpengaruh pada model belum tentu penyebab target.", ""),
            core.Finding("info", "Skor bergantung pada data", "Data kecil atau tidak representatif membuat skor tidak stabil. Lihat std antar fold.", ""),
            core.Finding("warn", "Validasi eksternal tetap perlu", "Sebelum dipakai untuk keputusan nyata, uji pada data baru dari periode/sumber berbeda.", ""),
            core.Finding("info", "Tidak ada tuning hiperparameter", "Semua model memakai pengaturan default yang masuk akal. Skrip reproduksi bisa dipakai untuk tuning lanjutan.", ""),
        ])

_exp = st.session_state.get("exp")
stepper_slot.markdown(ui.stepper_html(4 if _exp else 1), unsafe_allow_html=True)
ui.footnote("Catatan: hasil model bergantung pada kualitas, ukuran, representativitas, dan definisi target dataset. "
            "Periksa data leakage, ketidakseimbangan kelas, dan validasi eksternal sebelum menarik kesimpulan.")
