import os
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from itertools import product
from joblib import Parallel, delayed

# ✅ Tham số mô hình
INF = 10000  # Ngưỡng trạng thái lớn nhất được phép
MAX_ITER = 50  # Số lần lặp tối đa để mô hình hội tụ
TOL = 1e-4  # Sai số hội tụ

# ✅ Tập tham số cần tìm kiếm
EPSILON_list = [0.05, 0.1, 0.2]
DELTA_list = [0.1, 0.2, 0.4]
N_BETA_list = [2, 3, 4]

# ✅ Đường dẫn dữ liệu
input_folder = "data_2"  # Folder chứa các file mạng
truth_file = "HGRN.csv"  # File chứa danh sách gene đúng (1 cột tên 'Gene')
detail_output_folder = "../grid_detail"  # ✅ Thư mục lưu chi tiết kết quả
os.makedirs(detail_output_folder, exist_ok=True)

# ✅ Hàm đọc mạng từ file .txt
def import_network(file_path):
    with open(file_path, "r") as f:
        data = f.readlines()[1:]
    G = nx.DiGraph()
    for line in data:
        from_node, to_node, direction, weight = line.strip().split("\t")
        direction = int(direction)
        weight = float(weight)
        G.add_edge(from_node, to_node, weight=float(weight))
        if direction == 0:
            G.add_edge(to_node, from_node, weight=float(weight))
    return G

# ✅ Hàm xây dựng ma trận kề và danh sách láng giềng
def build_adjacency(G, node_order):
    n = len(node_order)
    node_index = {node: i for i, node in enumerate(node_order)}
    A = np.zeros((n, n))
    neighbors = {i: [] for i in range(n)}
    for u, v, data in G.edges(data=True):
        i, j = node_index[u], node_index[v]
        A[i, j] += data.get("weight", 1.0)
        neighbors[j].append(i)
    return A, neighbors, node_index

# ✅ Hàm cập nhật trạng thái của toàn bộ mạng tại mỗi bước lặp
def update_states(x, A, neighbors, beta_indices, beta_weights, fixed_nodes, EPSILON, DELTA):
    n = len(x)
    x_new = x.copy()
    for u in range(n):
        if u in fixed_nodes:
            continue
        influence = EPSILON * sum(A[v, u] * (x[v] - x[u]) for v in neighbors[u])
        beta_influence = DELTA * sum(
            w * (x[b] - x[u]) for b, w in zip(beta_indices, beta_weights[u])
        )
        x_new[u] = x[u] + influence + beta_influence
    return np.clip(x_new, -INF, INF)

# ✅ Mô phỏng mô hình với nhóm Beta gắn vào target_node
def simulate_one_target(G, beta_nodes, target_node, x_prev, alpha_idx, node_order, EPSILON, DELTA):
    all_nodes = node_order + [f"Beta{i}" for i in range(len(beta_nodes))]
    A, neighbors, node_index = build_adjacency(G, all_nodes)
    n = len(all_nodes)
    if x_prev.shape[0] != n:
        x_prev = np.pad(x_prev, (0, n - len(x_prev)), mode="constant")
    x = x_prev.copy()
    beta_indices = []
    fixed_nodes = set()
    beta_weights = [[0] * len(beta_nodes) for _ in range(n)]

    for i, beta in enumerate(beta_nodes):
        beta_name = f"Beta{i}"
        beta_idx = node_index[beta_name]
        A[beta_idx, node_index[target_node]] = 1.0
        neighbors[node_index[target_node]].append(beta_idx)
        x[beta_idx] = -1
        beta_indices.append(beta_idx)
        fixed_nodes.add(beta_idx)
        beta_weights[node_index[target_node]][i] = 1.0

    for _ in range(MAX_ITER):
        x_new = update_states(x, A, neighbors, beta_indices, beta_weights, fixed_nodes, EPSILON, DELTA)
        if np.linalg.norm(x_new - x) < TOL:
            break
        x = x_new

    return x[:len(G.nodes())]  # Trả về trạng thái của các node thường

# ✅ Hàm tính tổng hỗ trợ của một alpha node từ vector trạng thái
def compute_total_support(x_state, alpha_idx):
    return sum(1 if x > 0 else -1 if x < 0 else 0 for i, x in enumerate(x_state) if i != alpha_idx)

# ✅ Xử lý song song từng alpha node

def process_alpha(alpha_node, G, node_order, N_BETA, EPSILON, DELTA):
    alpha_idx = node_order.index(alpha_node)
    x_state = np.zeros(len(node_order))
    x_state[alpha_idx] = 1
    for target_node in node_order:
        if target_node == alpha_node:
            continue
        beta_nodes = node_order[:N_BETA]
        x_state = simulate_one_target(G, beta_nodes, target_node, x_state, alpha_idx, node_order, EPSILON, DELTA)
    support = compute_total_support(x_state, alpha_idx)
    return alpha_node, support

# ✅ Mô phỏng toàn bộ mạng với một tổ hợp tham số (song song hoá alpha nodes)
def run_simulation(G, EPSILON, DELTA, N_BETA, save_detail=False, detail_file=None):
    node_order = list(G.nodes())
    results = Parallel(n_jobs=-1)(
        delayed(process_alpha)(alpha_node, G, node_order, N_BETA, EPSILON, DELTA)
        for alpha_node in tqdm(node_order, desc=f"  ↳ Alpha nodes")
    )
    support_dict = dict(results)
    if save_detail and detail_file:
        pd.DataFrame(results, columns=["Alpha_Node", "Total_Support"]).to_csv(detail_file, index=False)
    top_100_nodes = sorted(support_dict.items(), key=lambda x: x[1], reverse=True)[:100]
    return set(node for node, _ in top_100_nodes)

# ✅ Chạy toàn bộ grid search và ghi kết quả

def main():
    truth_genes = set(pd.read_csv(truth_file)["Gene"].astype(str))
    files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]
    records = []

    for EPSILON, DELTA, N_BETA in product(EPSILON_list, DELTA_list, N_BETA_list):
        print(f"▶️ EPS={EPSILON}, DELTA={DELTA}, N_BETA={N_BETA}")
        for file in files:
            print(f"  🔄 Đang xử lý file: {file}")
            G = import_network(os.path.join(input_folder, file))
            detail_filename = f"{file.replace('.txt', '')}_eps{EPSILON}_del{DELTA}_nb{N_BETA}.csv"
            detail_path = os.path.join(detail_output_folder, detail_filename)
            top_genes = run_simulation(G, EPSILON, DELTA, N_BETA, save_detail=True, detail_file=detail_path)
            match = len(truth_genes & top_genes)
            records.append({
                "File": file,
                "EPSILON": EPSILON,
                "DELTA": DELTA,
                "N_BETA": N_BETA,
                "Matched_Genes": match,
                "Matching_Rate": match / 100
            })

    df = pd.DataFrame(records).sort_values("Matching_Rate", ascending=False)
    df.to_csv("grid_search_result.csv", index=False)
    print("✅ Đã lưu kết quả vào grid_search_result.csv")

if __name__ == "__main__":
    main()
