# 🔬 Multi-agent Outside Competitive Dynamics Model — Streamlit Interface

This project simulates external competitive dynamics on biological gene networks using Streamlit UI. It's designed for analyzing potential gene targets through competitive multi-agent simulation and matching with biological databases.

---

## 📦 Features
- Upload custom gene regulatory network `.txt`
- Run simulation with configurable parameters (epsilon, delta, N_BETA, etc.)
- Visualize network structure interactively
- Match results with **OncoKB** and **PubMed**
- Download simulation and matched results
- Temporary files cleaned up after use

---

## 📁 File Structure
```
App/
├── UI.py                 # Streamlit interface (this file)
Simulate/
├── Simulate_Model.py     # Core simulation model
functions/
├── Compare.py            # Gene matching (OncoKB / PubMed)
```

---

## ⚙️ Installation
```bash
# Clone repo
https://github.com/yourusername/multi-agent-competition-model.git
cd multi-agent-competition-model

# Setup environment
conda create -n beta_env python=3.8
conda activate beta_env

# Install requirements
pip install -r requirements.txt
```

---

## 🚀 Run Application
```bash
cd App
streamlit run UI.py
```
Then open: http://localhost:8501

---

## 📄 Input Format
Tab-separated `.txt` file with columns:
```
source\ttarget\tdirection\tweight
```
- `direction`: 0 = bidirectional, 1 = one-way edge

**Example:**
```
A\tB\t1\t0.8
B\tC\t0\t1.0
```

---

## 🧮 Parameters
- **Epsilon (ε):** strength of internal propagation
- **Delta (δ):** external Beta force
- **N_BETA:** number of Beta nodes added
- **Max Iter / Tolerance:** convergence settings

---

## 🧬 Biological Matching
Uses `functions/Compare.py` to cross-reference top predicted genes with:
- `OncoKB` (Cancer gene knowledge base)
- `PubMed` (Clinical gene publications)

---

## 📥 Output
- Results shown on screen and downloadable as `.csv`
- Matched results with biological databases

---

## 👤 Author
Developed by **Phạm Thành Bình** @HUST. For academic or research use.

Contact: https://github.com/thanhbinhnd2002
