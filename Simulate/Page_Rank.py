# ✅ PageRank.py
# Ý tưởng: Tính PageRank cho mỗi gene trong mạng lưới
# Đầu vào: File mạng lưới (dạng edgelist)
# Đầu ra: File CSV (pagerank_results.csv) với cột: Gene, PageRank

import sys
import pandas as pd
import networkx as nx

# Đọc tham số dòng lệnh
if len(sys.argv) != 2:
    print("Usage: python PageRank.py <network_file>")
    sys.exit(1)

network_file = sys.argv[1]

# Đọc mạng lưới
G = nx.read_edgelist(network_file, delimiter="\t", create_using=nx.DiGraph())

# Tính PageRank
pagerank = nx.pagerank(G)

# Tạo DataFrame kết quả
df = pd.DataFrame(pagerank.items(), columns=["Gene", "PageRank"])

# Lưu kết quả
df.to_csv("pagerank_results.csv", index=False)
print("✅ Đã lưu kết quả vào pagerank_results.csv")
