import sys
import os
import tempfile
import threading
import time
import streamlit as st
import pandas as pd

# Thêm thư mục để import các module phụ
sys.path.append(os.path.abspath("."))

from Simulate.Simulate_Model import import_network, simulate
from functions.Compare import match_with_oncokb_pubmed

st.set_page_config(page_title="Cancer Gene Simulation", layout="wide")
st.title("🔬 Multi-agent Outside Competitive Dynamics Model")

# --- Sidebar ---
st.sidebar.header("⚙️ Simulation Settings")
uploaded_file = st.sidebar.file_uploader("Upload a .txt network file", type=["txt"])
EPSILON = st.sidebar.slider("Epsilon", 0.05, 1.0, 0.1, step=0.01)
DELTA = st.sidebar.slider("Delta", 0.01, 1.0, 0.2, step=0.01)
MAX_ITER = st.sidebar.number_input("Max Iterations", 10, 200, 50)
TOL = st.sidebar.number_input("Tolerance", 1e-6, 1e-2, 1e-4, format="%e")
N_BETA = st.sidebar.slider("Number of Beta per group", 1, 10, 2)
start = st.sidebar.button("🚀 Run Simulation", disabled=(uploaded_file is None))

# --- Reset khi đổi file ---
if uploaded_file:
    filename = uploaded_file.name
    if st.session_state.get("filename") != filename:
        for key in ["result_df", "matched_df", "out_file", "running", "thread", "simulation_result", "stop_signal"]:
            st.session_state.pop(key, None)

    st.session_state["filename"] = filename
    st.code(filename)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="wb") as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = tmp.name
        st.session_state["temp_path"] = temp_path

    G = import_network(temp_path)
    st.write(f"✅ Network loaded with **{len(G.nodes())} nodes** and **{len(G.edges())} edges**.")
else:
    st.warning("⚠️ Please upload a network file.")

# --- Nhận kết quả từ mô phỏng ---
def receive_simulation_result(out_file, df):
    st.session_state["simulation_result"] = {
        "out_file": out_file,
        "result_df": df
    }

# --- Thread thực thi mô phỏng ---
def run_simulation_safe(temp_path, eps, delta, max_iter, tol, n_beta):
    from Simulate import Simulate_Model
    result = Simulate_Model.simulate(
        file_path=temp_path,
        EPSILON=eps,
        DELTA=delta,
        MAX_ITER=max_iter,
        TOL=tol,
        N_BETA=n_beta,
        output_folder="Output",
        stop_check_func=lambda: st.session_state.get("stop_signal", False)
    )
    if result is not None:
        out_file, df = result
        receive_simulation_result(out_file, df)
    st.session_state["running"] = False
    try:
        os.remove(temp_path)
    except:
        pass

# --- Bắt đầu mô phỏng ---
if start and "temp_path" in st.session_state:
    if not st.session_state.get("running", False):
        st.session_state["running"] = True
        st.session_state["stop_signal"] = False
        temp_path = st.session_state["temp_path"]  # ✅ truyền temp_path vào thread
        thread = threading.Thread(
            target=run_simulation_safe,
            args=(temp_path, EPSILON, DELTA, MAX_ITER, TOL, N_BETA)
        )
        thread.start()
        st.session_state["thread"] = thread

# --- Nếu đang mô phỏng
if st.session_state.get("running", False):
    st.info(f"⏳ Running simulation on: `{st.session_state['filename']}`")
    if st.button("🔴 Cancel Simulation"):
        st.session_state["stop_signal"] = True
        st.warning("⚠️ Stop signal sent to simulation.")

    with st.spinner("Simulation in progress..."):
        time.sleep(0.5)
        st.rerun()

# --- Hiển thị kết quả nếu có ---
if "simulation_result" in st.session_state:
    st.success("✅ Simulation completed.")
    st.subheader(f"📊 Simulation Result for: `{st.session_state['filename']}`")

    df = st.session_state["simulation_result"]["result_df"]
    out_file = st.session_state["simulation_result"]["out_file"]
    st.session_state["result_df"] = df
    st.session_state["out_file"] = out_file

    st.dataframe(df.sort_values("Total_Support", ascending=True))
    st.download_button(
        "⬇️ Download Result CSV",
        data=df.to_csv(index=False),
        file_name=os.path.basename(out_file),
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
