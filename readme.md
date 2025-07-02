# 🔬 Multi-agent Outside Competitive Dynamics Model — Streamlit Interface

This project simulates external competitive dynamics on biological gene networks using a Streamlit UI. It's designed for analyzing potential gene targets through competitive multi-agent simulation and matching with biological databases.

---

## 📆 Features

* Upload custom gene regulatory network `.txt`
* Run simulation with configurable parameters (`epsilon`, `delta`, `N_BETA`, etc.)
* Visualize network structure interactively
* Match results with **OncoKB** and **PubMed**
* Download simulation and matched results as `.csv`
* Temporary files cleaned up after use

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

### Step 1: Clone the repository

```bash
https://github.com/yourusername/multi-agent-competition-model.git
cd multi-agent-competition-model
```

### Step 2: Setup environment (recommended with Anaconda)

```bash
conda create -n beta_env python=3.8
conda activate beta_env
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

#### ⚠️ Windows users: fix `UnicodeDecodeError`

If you encounter:

```
UnicodeDecodeError: 'charmap' codec can't decode byte...
```

Fix:

* Open `requirements.txt` with Notepad++ / VSCode
* Save as `UTF-8 with BOM`
* Try running `pip install` again

---

## 🚀 Run Application

```bash
cd App
streamlit run UI.py
```

Then open: [http://localhost:8501](http://localhost:8501)

---

## 📄 Input Format

Upload a **tab-separated** `.txt` file with the following columns:

```
source\ttarget\tdirection\tweight
```

* `direction`: 0 = bidirectional, 1 = one-way edge

**Example:**

```
A\tB\t1\t0.8
B\tC\t0\t1.0
```

---

## 🧼 Parameters

* **Epsilon (ε):** strength of internal propagation
* **Delta (δ):** external Beta force
* **N\_BETA:** number of Beta nodes added
* **Max Iter / Tolerance:** convergence settings

---

## 🧬 Biological Matching

Uses `functions/Compare.py` to cross-reference top predicted genes with:

* **OncoKB**: Cancer gene knowledge base
* **PubMed**: Clinical gene publication evidence

---

## 📅 Output

* Results shown in an interactive table on-screen
* Downloadable as `.csv`
* Includes matched results with OncoKB / PubMed

---

## 👤 Author

Developed by **Phạm Thành Bình** @HUST. For academic and research use only.

Contact: [https://github.com/thanhbinhnd2002](https://github.com/thanhbinhnd2002)
