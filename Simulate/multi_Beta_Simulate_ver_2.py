# === THƯ VIỆN CẦN THIẾT ===
import os
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import cpu_count

# === THAM SỐ MÔ HÌNH ===
INF = 10000
EPSILON = 0.05  # Hệ số ảnh hưởng từ các node thường
DELTA = 0.1     # Hệ số ảnh hưởng từ các node Beta
MAX_ITER = 50   # Số vòng lặp tối đa để cập nhật trạng thái
TOL = 1e-4      # Ngưỡng hội tụ
N_BETA = 2      # Số lượng Beta node được gán mỗi lần

# === GIAI ĐOẠN 1: Đọc mạng từ file ===
def import_network(file_path):
    """
    Đọc mạng từ file .txt với định dạng: from, to, direction, weight
    direction = 0: hai chiều, direction = 1: một chiều
    """
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

# === GIAI ĐOẠN 2: Xây dựng ma trận kề và hàng xóm ===
def build_adjacency(G, node_order):
    """
    Tạo ma trận kề A và dict neighbors lưu các hàng xóm ngược (cạnh đi vào node).
    node_order dùng để đánh số chỉ mục node nhất quán.
    """
    n = len(node_order)
    node_index = {node: i for i, node in enumerate(node_order)}
    A = np.zeros((n, n))
    neighbors = {i: [] for i in range(n)}
    for u, v, data in G.edges(data=True):
        i, j = node_index[u], node_index[v]
        A[i, j] += data.get("weight", 1.0)
        neighbors[j].append(i)
    return A, neighbors, node_index

# === GIAI ĐOẠN 3: Cập nhật trạng thái cho tất cả node thường ===
def update_states_multi_beta(x, A, neighbors, beta_indices, beta_weights, fixed_nodes):
    """
    Cập nhật trạng thái của tất cả node trong mạng tại 1 thời điểm.
    Bao gồm ảnh hưởng nội mạng (từ hàng xóm) và ảnh hưởng từ các Beta.
    """
    n = len(x)
    x_new = x.copy()
    for u in range(n):
        if u in fixed_nodes:
            continue  # Node cố định (Alpha hoặc Beta) không cập nhật
        # Ảnh hưởng nội mạng từ các node hàng xóm
        influence = EPSILON * sum(A[v, u] * (x[v] - x[u]) for v in neighbors[u])
        # Ảnh hưởng từ các Beta node
        beta_influence = DELTA * sum(
            w * (x[b] - x[u]) for b, w in zip(beta_indices, beta_weights[u])
        )
        # Cập nhật trạng thái node u
        x_new[u] = x[u] + influence + beta_influence
    return np.clip(x_new, -1000, 1000)  # Giới hạn để tránh tràn số

# === GIAI ĐOẠN 4: Gán tất cả Beta vào 1 target node và cập nhật trạng thái ===
def simulate_beta_on_target(G, beta_nodes, target_node, x_prev=None, alpha_idx=None, node_order=None):
    """
    Mô phỏng quá trình cạnh tranh: gán tất cả Beta vào 1 target node và lan truyền.
    """
    if node_order is None:
        node_order = list(G.nodes())

    # Tạo danh sách tất cả node (gốc + Beta)
    all_nodes = node_order + [f"Beta{i}" for i in range(len(beta_nodes))]
    A, neighbors, node_index = build_adjacency(G, all_nodes)
    n = len(all_nodes)

    # Khởi tạo trạng thái nếu chưa có
    if x_prev is None:
        x_prev = np.zeros(n)
        if alpha_idx is not None:
            alpha_node_name = node_order[alpha_idx]
            alpha_idx_in_new = node_index[alpha_node_name]
            x_prev[alpha_idx_in_new] = 1

    # Đảm bảo kích thước x_prev phù hợp
    if x_prev.shape[0] != n:
        x_prev = np.pad(x_prev, (0, n - x_prev.shape[0]), mode="constant")

    x = x_prev.copy()
    beta_indices = []
    fixed_nodes = set()
    beta_weights = [[0] * len(beta_nodes) for _ in range(n)]

    # Gán tất cả Beta vào cùng 1 target node
    for i, beta in enumerate(beta_nodes):
        beta_name = f"Beta{i}"
        beta_idx = node_index[beta_name]
        A[beta_idx, node_index[target_node]] = 1.0
        neighbors[node_index[target_node]].append(beta_idx)
        x[beta_idx] = -1
        beta_indices.append(beta_idx)
        fixed_nodes.add(beta_idx)
        beta_weights[node_index[target_node]][i] = 1.0

    # Cập nhật trạng thái cho đến khi hội tụ
    for _ in range(MAX_ITER):
        x_new = update_states_multi_beta(x, A, neighbors, beta_indices, beta_weights, fixed_nodes)
        if np.linalg.norm(x_new - x) < TOL:
            break
        x = x_new

    return x[:len(G.nodes())]  # Trả về trạng thái của node thường (loại bỏ Beta)

# === GIAI ĐOẠN 5: Tính tổng hỗ trợ cho Alpha node ===
def compute_total_support(x_state, alpha_idx):
    """
    Tổng hỗ trợ = số node ủng hộ (x > 0) trừ số node phản đối (x < 0), không tính Alpha.
    """
    return sum(1 if x > 0 else -1 if x < 0 else 0 for i, x in enumerate(x_state) if i != alpha_idx)

# === GIAI ĐOẠN 6: Xử lý 1 node Alpha ===
def process_alpha(alpha_node, G, beta_nodes):
    """
    Với mỗi node Alpha:
        - Gán tất cả Beta vào từng target node còn lại
        - Mỗi lần gán, cập nhật trạng thái toàn mạng
        - Cuối cùng tính tổng hỗ trợ cho Alpha
    """
    node_order = list(G.nodes())
    alpha_idx = node_order.index(alpha_node)
    x_state = np.zeros(len(node_order))
    x_state[alpha_idx] = 1

    for target_node in node_order:
        if target_node == alpha_node:
            continue
        x_state = simulate_beta_on_target(G, beta_nodes, target_node, x_state, alpha_idx, node_order)

    support = compute_total_support(x_state, alpha_idx)
    return {"Alpha_Node": alpha_node, "Total_Support": support}

# === GIAI ĐOẠN 7: MAIN PROGRAM ===
def main():
    input_folder = "data_1"
    # Thư mục lưu kết quả theo cấu hình mô hình
    output_folder = f"Output_test/INF{INF}_EPS{EPSILON}_DELTA{DELTA}_ITER{MAX_ITER}_TOL{TOL}_NBETA{N_BETA}"
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if not file.endswith(".txt"):
            continue
    
        path = os.path.join(input_folder, file)
        G = import_network(path)
    
        all_nodes = list(G.nodes())

        # Chạy song song cho tất cả node Alpha
        results = Parallel(n_jobs=-1)(
            delayed(process_alpha)(alpha_node, G, all_nodes[:N_BETA])
            for alpha_node in tqdm(all_nodes, desc=f"🔁 Xử lý file {file}")
        )

        # Ghi kết quả ra file CSV
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(output_folder, file.replace(".txt", ".csv")), index=False)
        print(f"✅ Xong: {file}")

# === ĐIỂM KHỞI ĐẦU CHƯƠNG TRÌNH ===
if __name__ == "__main__":
    main()
