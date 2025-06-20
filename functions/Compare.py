# ✅ compare.py — chỉnh đường dẫn tuyệt đối cho các file dữ liệu sinh học và hỗ trợ lọc + lưu

import pandas as pd
import os

# ✅ Lấy thư mục gốc nơi đặt file script đang chạy
ROOT = os.path.dirname(os.path.abspath(__file__))

# ✅ Các đường dẫn dữ liệu tương đối từ thư mục gốc project
ONCOKB_PATH = os.path.join(ROOT, "..", "Cancer gene OncoKB30012025.xlsx")
PUBMED_PATH = os.path.join(ROOT, "..", "Clinical.xlsx")
MART_PATH = os.path.join(ROOT, "..", "mart_biotool.txt")

# ✅ Hàm chính để gọi từ app — cho phép lọc top_n và lưu file kết quả nếu muốn

def match_with_oncokb_pubmed(model_result: pd.DataFrame, top_n: int = None, save_path: str = None) -> pd.DataFrame:
    oncokb = pd.read_excel(ONCOKB_PATH)
    pubmed = pd.read_excel(PUBMED_PATH)
    mart = pd.read_csv(MART_PATH, sep="\t")

    def get_pubmed_info(gene):
        symbol_match = pubmed[pubmed['Symbol'] == gene]
        alias_match = pubmed[pubmed['Alias symbol'].fillna('').str.split(', ').apply(lambda x: gene in x)]
        if not symbol_match.empty:
            return symbol_match.iloc[0]['PubmedID']
        elif not alias_match.empty:
            return alias_match.iloc[0]['PubmedID']
        else:
            return ""

    def get_ensembl_id(gene, aliases):
        symbol_match = pubmed[pubmed['Symbol'] == gene]
        alias_match = pubmed[pubmed['Alias symbol'].fillna('').str.split(', ').apply(lambda x: gene in x)]
        if not symbol_match.empty:
            return symbol_match.iloc[0]['Ensembl ID']
        elif not alias_match.empty:
            return alias_match.iloc[0]['Ensembl ID']
        else:
            for name in [gene] + aliases:
                mart_match = mart[mart['Gene name'] == name]
                if not mart_match.empty:
                    return mart_match.iloc[0]['Gene stable ID']
            return ""

    def check_oncokb(gene):
        symbol_match = oncokb[oncokb['Hugo Symbol'] == gene]
        alias_match = oncokb[oncokb['Gene Aliases'].fillna('').str.split(', ').apply(lambda x: gene in x)]
        if not symbol_match.empty:
            row = symbol_match.iloc[0]
            return row['Hugo Symbol'], row['Gene Aliases'], row['Is Oncogene'], row['Is Tumor Suppressor Gene'], True
        elif not alias_match.empty:
            row = alias_match.iloc[0]
            return row['Hugo Symbol'], row['Gene Aliases'], row['Is Oncogene'], row['Is Tumor Suppressor Gene'], True
        else:
            return gene, "", "", "", False

    output = []
    for _, row in model_result.iterrows():
        gene = row['Alpha_Node'] if 'Alpha_Node' in row else (row['Symbol'] if 'Symbol' in row else None)
        total_support = row['Total_Support']

        if gene is None:
            continue

        symbol, alias, is_oncogene, is_tsg, in_oncokb = check_oncokb(gene)
        pubmed_id = get_pubmed_info(gene)
        ensembl_id = get_ensembl_id(symbol, alias.split(', ') if isinstance(alias, str) else [])

        output.append({
            "Ensembl ID": ensembl_id,
            "Symbol": symbol,
            "Alias symbol": alias,
            "Total_Support": total_support,
            "Is Oncogene": is_oncogene,
            "Is Tumor Suppressor Gene": is_tsg,
            "In OnkoKB": in_oncokb,
            "PubmedID": pubmed_id
        })

    matched_df = pd.DataFrame(output)
    if top_n is not None:
        matched_df = matched_df.sort_values(by="Total_Support", ascending=False).head(top_n)

    if save_path is not None:
        matched_df.to_csv(save_path, index=False)

    return matched_df
