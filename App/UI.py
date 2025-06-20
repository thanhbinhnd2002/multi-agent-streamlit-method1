import sys
import os
import tempfile
import streamlit as st
import pandas as pd

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

# --- Upload + preview ---
if uploaded_file:
    filename = uploaded_file.name
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

# --- Run simulation ---
if start and "temp_path" in st.session_state:
    with st.spinner("Running simulation..."):
        output_folder = "Output"
        out_file = simulate(
            file_path=st.session_state["temp_path"],
            EPSILON=EPSILON,
            DELTA=DELTA,
            MAX_ITER=MAX_ITER,
            TOL=TOL,
            N_BETA=N_BETA,
            output_folder=output_folder
        )
        try:
            os.remove(st.session_state["temp_path"])
        except:
            pass
        st.session_state["out_file"] = out_file
        st.session_state["result_df"] = pd.read_csv(out_file)

# --- Show result ---
if "result_df" in st.session_state and "filename" in st.session_state:
    st.success("✅ Simulation completed.")
    st.subheader(f"📊 Simulation Result for: `{st.session_state['filename']}`")

    df = st.session_state["result_df"]
    st.dataframe(df.sort_values("Total_Support", ascending=True))
    st.download_button(
        "⬇️ Download Result CSV",
        data=df.to_csv(index=False),
        file_name=os.path.basename(st.session_state["out_file"]),
        mime="text/csv"
    )

    if st.button("🔍 Đối chiếu với OncoKB và PubMed"):
        matched_df = match_with_oncokb_pubmed(df)
        st.session_state["matched_df"] = matched_df

# --- Matched result ---
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
