import os
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from tqdm import tqdm

EPSILON = 0.1
DELTA = 0.2
MAX_ITER = 50
TOL = 1e-4
N_BETA = 2

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

def build_adjacency(G, node_order):
    n = len(node_order)
    node_index = {node: i for i, node in enumerate(node_order)}
    A = np.zeros((n, n))
    neighbors = {i: [] for i in range(n)}
    for u, v, data in G.edges(data=True):
        i, j = node_index[u], node_index[v]
        A[i, j] += float(data.get("weight", 1.0))
        neighbors[j].append(i)
    return A, neighbors, node_index

def update_states(x, A, neighbors, beta_indices, beta_weights, fixed_nodes, EPSILON, DELTA, MAX_ITER, TOL):
    for _ in range(MAX_ITER):
        x_new = x.copy()
        for u in range(len(x)):
            if u in fixed_nodes:
                continue
            influence = EPSILON * sum(A[v, u] * (x[v] - x[u]) for v in neighbors[u])
            beta_influence = DELTA * sum(w * (x[b] - x[u]) for b, w in zip(beta_indices, beta_weights[u]))
            x_new[u] = x[u] + influence + beta_influence
        if np.linalg.norm(x_new - x) < TOL:
            break
        x = x_new
    return np.clip(x, -1000, 1000)

def simulate_beta_on_target(G, beta_nodes, target_node, x_prev, alpha_idx, node_order,
                            EPSILON, DELTA, MAX_ITER, TOL):
    all_nodes = node_order + [f"Beta{i}" for i in range(len(beta_nodes))]
    A, neighbors, node_index = build_adjacency(G, all_nodes)
    n = len(all_nodes)

    if x_prev.shape[0] != n:
        x_prev = np.pad(x_prev, (0, n - x_prev.shape[0]), mode="constant")

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

    x = update_states(x, A, neighbors, beta_indices, beta_weights, fixed_nodes, EPSILON, DELTA, MAX_ITER, TOL)
    return x[:len(node_order)]

def compute_total_support(x_state, alpha_idx):
    return sum(1 if x > 0 else -1 if x < 0 else 0 for i, x in enumerate(x_state) if i != alpha_idx)

def simulate_alpha(alpha_node, G, beta_nodes, EPSILON, DELTA, MAX_ITER, TOL):
    node_order = list(G.nodes())
    alpha_idx = node_order.index(alpha_node)
    x_state = np.zeros(len(node_order))
    x_state[alpha_idx] = 1

    for target_node in node_order:
        if target_node == alpha_node:
            continue
        x_state = simulate_beta_on_target(G, beta_nodes, target_node, x_state, alpha_idx, node_order,
                                          EPSILON, DELTA, MAX_ITER, TOL)

    support = compute_total_support(x_state, alpha_idx)
    return {"Alpha_Node": alpha_node, "Total_Support": support}

def simulate(file_path, EPSILON=0.1, DELTA=0.2, MAX_ITER=50, TOL=1e-4, N_BETA=2):
    G = import_network(file_path)
    all_nodes = list(G.nodes())
    beta_nodes = all_nodes[:N_BETA]

    results = Parallel(n_jobs=cpu_count() // 2)(
        delayed(simulate_alpha)(alpha, G, beta_nodes, EPSILON, DELTA, MAX_ITER, TOL)
        for alpha in tqdm(all_nodes, desc="🔁 Alpha nodes")
    )

    df = pd.DataFrame(results).sort_values(by="Total_Support", ascending=False)
    return df

