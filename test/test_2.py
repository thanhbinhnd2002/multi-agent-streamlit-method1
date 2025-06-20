# ✅ Ý tưởng:
# - Đọc file mạng và file driver-target node (pairs.csv)
# - Với mỗi cặp driver-target, chạy mô hình cạnh tranh ngoài:
#   + Alpha nodes: driver nodes (từng node 1)
#   + Beta nodes: target nodes (gán toàn bộ cùng lúc)
# - Tính tổng hỗ trợ của từng alpha node

import os
import networkx as nx
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count
import numpy as np
from ast import literal_eval

INF = 10000
EPSILON = 0.1
DELTA = 0.2
MAX_ITER = 200
TOL = 1e-6

# ✅ Đọc mạng từ file
def import_network(file_path):
    with open(file_path, "r") as f:
        data = f.readlines()[1:]
    G = nx.DiGraph()
    for line in data:
        from_node, to_node, direction, weight = line.strip().split("\t")
        direction = int(direction)
        weight = float(weight)
        G.add_edge(from_node, to_node, weight=weight)
        if direction == 0:
            G.add_edge(to_node, from_node, weight=weight)
    return G

# ✅ Tạo ma trận kề và hàng xóm
def build_adjacency(G, node_order):
    n = len(node_order)
    node_index = {node: i for i, node in enumerate(node_order)}
    A = np.zeros((n, n))
    neighbors = {i: [] for i in range(n)}
    for u, v, data in G.edges(data=True):
        if u not in node_index or v not in node_index:
            continue
        i, j = node_index[u], node_index[v]
        A[i, j] += data.get("weight", 1.0)
        neighbors[j].append(i)
    return A, neighbors, node_index

# ✅ Cập nhật trạng thái có kiểm tra
def update_states_multi_beta(x, A, neighbors, beta_indices, beta_weights, fixed_nodes):
    n = len(x)
    x_new = x.copy()
    for u in range(n):
        if u in fixed_nodes:
            continue
        influence = 0.0
        beta_influence = 0.0
        for v in neighbors[u]:
            if np.isnan(x[v]) or np.isnan(x[u]):
                continue
            influence += A[v, u] * (x[v] - x[u])
        for b, w in zip(beta_indices, beta_weights[u]):
            if np.isnan(x[b]) or np.isnan(x[u]):
                continue
            beta_influence += w * (x[b] - x[u])
        x_new[u] = x[u] + EPSILON * influence + DELTA * beta_influence

        # ✅ Chặn tràn số
        if np.isnan(x_new[u]) or np.isinf(x_new[u]):
            print(f"⚠️ Trạng thái không hợp lệ tại node {u}: {x_new[u]} → đặt lại 0")
            x_new[u] = 0.0
        else:
            x_new[u] = max(-INF, min(INF, x_new[u]))
    return x_new

# ✅ Mô phỏng cạnh tranh ngoài
def simulate_competition(G, alpha_nodes, beta_nodes):
    node_order = list(G.nodes()) + [f"Beta{i}" for i in range(len(beta_nodes))]
    A, neighbors, node_index = build_adjacency(G, node_order)
    n = len(node_order)

    x = np.zeros(n)
    alpha_indices = [node_index[a] for a in alpha_nodes if a in node_index]
    for idx in alpha_indices:
        x[idx] = 1.0

    beta_indices = []
    fixed_nodes = set()
    beta_weights = [[0] * len(beta_nodes) for _ in range(n)]

    for i, attach_node in enumerate(beta_nodes):
        if attach_node not in node_index:
            print(f"⚠️ Node {attach_node} không tồn tại trong mạng. Bỏ qua.")
            continue
        beta_name = f"Beta{i}"
        beta_idx = node_index[beta_name]
        A[beta_idx, node_index[attach_node]] = 1.0
        neighbors[node_index[attach_node]].append(beta_idx)
        x[beta_idx] = -1.0
        beta_indices.append(beta_idx)
        fixed_nodes.add(beta_idx)
        beta_weights[node_index[attach_node]][i] = 1.0

    for _ in range(MAX_ITER):
        x_new = update_states_multi_beta(x, A, neighbors, beta_indices, beta_weights, fixed_nodes)
        if np.linalg.norm(x_new - x) < TOL:
            break
        x = x_new

    return x[:len(G.nodes())]  # Trạng thái các node thường

# ✅ Tính tổng hỗ trợ
def compute_total_support(x_state, alpha_indices):
    support_dict = {}
    for alpha_idx in alpha_indices:
        support = 0
        for j in range(len(x_state)):
            if j == alpha_idx:
                continue
            if np.isnan(x_state[j]):
                continue
            if x_state[j] > 0:
                support += 1
            elif x_state[j] < 0:
                support -= 1
        support_dict[alpha_idx] = support
    return support_dict

# ✅ Hàm xử lý từng bản ghi driver-target
def process_driver_target(G, drivers, targets):
    node_order = list(G.nodes())
    results = []
    for alpha in drivers:
        if alpha not in node_order:
            continue
        x_state = simulate_competition(G, [alpha], targets)
        alpha_idx = node_order.index(alpha)
        support = compute_total_support(x_state, [alpha_idx])
        results.append({"Alpha_Node": alpha, "Total_Support": support[alpha_idx]})
    return results

# --- MAIN ---
if __name__ == "__main__":
    input_folder = "data_4"
    pair_folder = "driver_nodes_2"
    output_folder = "output_multi_beta_pair"
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if not file.endswith(".txt"):
            continue
        path = os.path.join(input_folder, file)
        base = os.path.splitext(file)[0]
        G = import_network(path)
        pair_file = os.path.join(pair_folder, f"{base}_pairs.csv")
        pairs_df = pd.read_csv(pair_file)

        results = Parallel(n_jobs=cpu_count() // 2)(
            delayed(process_driver_target)(
                G,
                literal_eval(row["Driver_Nodes"]),
                literal_eval(row["Target_Nodes"])
            )
            for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc=f"🔁 {base}")
        )

        all_results = [item for sublist in results for item in sublist]
        df = pd.DataFrame(all_results)
        out_file = os.path.join(output_folder, base + "_new2.csv")
        df.to_csv(out_file, index=False)
        print(f"✅ Đã lưu kết quả vào {out_file}")
