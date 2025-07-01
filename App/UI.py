# ✅ UI.py — Streamlit interface with visualization and result matching support

import sys
import os
import tempfile
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append(os.path.abspath(".."))
from Simulate.Simulate_Model import import_network, simulate
from functions.Compare import match_with_oncokb_pubmed

# --- Cấu hình Streamlit ---
st.set_page_config(page_title="Cancer Gene Simulation", layout="wide")
st.title("🔬 Multi-agent Outside Competitive Dynamics Model")

# --- Sidebar: tải file và chọn tham số ---
st.sidebar.header("⚙️ Simulation Settings")
uploaded_file = st.sidebar.file_uploader("Upload a .txt network file", type=["txt"])
EPSILON = st.sidebar.slider("Epsilon", 0.05, 1.0, 0.1, step=0.01)
DELTA = st.sidebar.slider("Delta", 0.01, 1.0, 0.2, step=0.01)
MAX_ITER = st.sidebar.number_input("Max Iterations", 10, 200, 50)
TOL = st.sidebar.number_input("Tolerance", 1e-6, 1e-2, 1e-4, format="%e")
N_BETA = st.sidebar.slider("Number of Beta per group", 1, 10, 2)

start = st.sidebar.button("🚀 Run Simulation", disabled=(uploaded_file is None))
draw = st.sidebar.button("🖼️ Draw Network", disabled=(uploaded_file is None))

# --- Đọc file mạng ---
if uploaded_file:
    filename = uploaded_file.name
    st.session_state["filename"] = filename
    st.code(filename)

    # ✅ Ghi file tạm đúng chuẩn trong môi trường cloud
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, filename)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.session_state["temp_path"] = temp_path

    G = import_network(temp_path)
    st.session_state["graph"] = G
    st.write(f"✅ Network loaded with **{len(G.nodes())} nodes** and **{len(G.edges())} edges**.")
else:
    st.warning("⚠️ Please upload a network file.")

# --- Vẽ mạng ---
if draw and "graph" in st.session_state:
    G = st.session_state["graph"]
    fig, ax = plt.subplots(figsize=(7, 5))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=False, node_size=30, edge_color="gray", ax=ax)
    st.pyplot(fig)

# --- Chạy mô phỏng ---
if start and "temp_path" in st.session_state:
    with st.spinner("Running simulation..."):
        result = simulate(
            file_path=st.session_state["temp_path"],
            EPSILON=EPSILON,
            DELTA=DELTA,
            MAX_ITER=MAX_ITER,
            TOL=TOL,
            N_BETA=N_BETA
        )

        # ✅ Nếu simulate trả về tuple → lấy phần tử đầu
        if isinstance(result, tuple):
            df = result[0]
        else:
            df = result

        st.session_state["result_df"] = df

# --- Hiển thị kết quả ---
if "result_df" in st.session_state and "filename" in st.session_state:
    df = st.session_state["result_df"]

    if df is None:
        st.error("❌ Kết quả mô phỏng là None. Mô phỏng đã thất bại.")
        st.stop()

    if isinstance(df, tuple):
        df = df[0]

    if df.empty:
        st.error("⚠️ Kết quả mô phỏng rỗng. Không có node nào được xử lý.")
        st.stop()

    if "Total_Support" not in df.columns:
        st.error("⚠️ Thiếu cột 'Total_Support' trong kết quả. Kiểm tra lại hàm simulate().")
        st.write(df)
        st.stop()

    st.success("✅ Simulation completed.")
    st.subheader(f"📊 Simulation Result for: `{st.session_state['filename']}`")
    st.dataframe(df.sort_values("Total_Support", ascending=True))
    df = st.session_state["result_df"]


    st.download_button(
        "⬇️ Download Result CSV",
        data=df.to_csv(index=False),
        file_name="simulation_result.csv",
        mime="text/csv"
    )
    if st.button("🔍 Đối chiếu với OncoKB và PubMed"):
        matched_df = match_with_oncokb_pubmed(df)
        st.session_state["matched_df"] = matched_df

# --- Kết quả đối chiếu ---
if "matched_df" in st.session_state:
    st.subheader("🧬 Matched Genes (OncoKB / PubMed)")
    matched_df = st.session_state["matched_df"]
    st.dataframe(matched_df)
    st.download_button(
        "💾 Tải kết quả đối chiếu",
        data=matched_df.to_csv(index=False),
        file_name="matched_result.csv",
        mime="text/csv"
    )
