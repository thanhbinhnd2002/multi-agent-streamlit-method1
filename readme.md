# Multi-agent Outside Competitive Dynamics Model

This project implements a **multi-agent outside competitive dynamics model** to simulate how external agents (e.g. drugs) influence genes in a biological network, with the goal of identifying potential **cancer target genes**.

The project is based on the simulation script:
→ `Simulate/multi_Beta_Simulate_ver_2.py`

---

## 📁 Folder Structure

```
.
├── App/                # (Optional) UI or additional tools
├── Data/               # Input biological network files (.txt)
├── functions/          # Utility scripts for gene mapping, OncoKB/PubMed matching
├── Output/             # Simulation results will be saved here
├── Results/            # Matched results and ranked gene tables
├── Simulate/           # Main simulation script: multi_Beta_Simulate_ver_2.py
├── Test/               # Testing or experimental scripts
├── Cancer gene OncoKB30012025.xlsx  # External validation dataset
├── Clinical.xlsx                       # Additional biological data
├── HGRN.csv                            # Example gene regulatory network
├── mart_biotool.txt                    # Mapping Ensembl ID ↔ Gene symbol
├── readme.md                           # ← This file
├── requirements.txt                    # Python dependencies
```

---

## 🛠️ Installation

Use Python 3.8+ and install required packages:

```bash
pip install -r requirements.txt
```

Basic requirements:

```
numpy
pandas
networkx
tqdm
joblib
```

---

## ▶️ How to Run the Model

### Step 1: Prepare Input Network

* Place your `.txt` input networks in the `Data/` folder.
* File format (tab-separated):

  ```
  Source<TAB>Target<TAB>Direction<TAB>Weight
  ```

  * `Direction = 1`: one-way edge
  * `Direction = 0`: bidirectional edge

### Step 2: Run Simulation

```bash
python Simulate/multi_Beta_Simulate_ver_2.py
```

This script will:

* Simulate the spread of influence using a multi-agent competitive dynamics model.
* Compute the **Total Support** score for each node (gene).
* Save output to: `Output/`

### Step 3: Interpret Results

Each output CSV file will contain:

| Alpha\_Node | Total\_Support |
| ----------- | -------------- |
| SMAD3       | -898           |
| TP53        | -722           |

* The higher the absolute Total Support, the more strongly supported (or opposed) a gene is under competition.
* You can match top-ranking genes with datasets in `Results/` using tools in `functions/`.

---

## 🥺 Example

Input:
→ `Data/Human Gene Regulatory Network - Input.txt`

Output:
→ `Output/Human Gene Regulatory Network - Input.csv`

---

## 📚 Reference

This project is part of a graduation thesis:

> **“Ứng dụng mô hình động lực học cạnh tranh ngoài đa tác nhân để dự đoán gene mục tiêu điều trị ung thư”**
> Phạm Thanh Bình, HUST 2025.

---

## 👨‍💼 Author

* **Pham Thanh Binh**
  Email: [binhpt207587@sis.hust.edu.vn](mailto:binhpt207587@sis.hust.edu.vn)
  Hanoi University of Science and Technology
  Supervisor: Assoc. Prof. Phạm Văn Hải
