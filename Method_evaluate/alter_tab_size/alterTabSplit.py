# ALTER 식 분류 + 세분화된 셀 카운트 버킷
import os
import re
import pandas as pd

TABLE_ROOT = "/home/wooo519/tableReasoningFinal/WikiTableQuestions"
OUT_DIR = "./tabsize_full/"

def read_tsv_by_tabcount(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    if not lines:
        return pd.DataFrame()

    # 2번째 줄 마크다운 정렬줄 제거
    if len(lines) >= 2:
        md_cells = lines[1].split("\t")
        if md_cells and all(re.fullmatch(r"[-:\s]+", (c or "").strip()) for c in md_cells):
            del lines[1]

    header = lines[0].split("\t")
    ncols = len(header)
    rows = []
    buf = ""

    for raw in lines[1:]:
        buf = (buf + "\n" + raw) if buf else raw
        if buf.count("\t") == ncols - 1:
            cells = buf.split("\t")
            if len(cells) < ncols:
                cells += [""] * (ncols - len(cells))
            elif len(cells) > ncols:
                cells = cells[:ncols-1] + ["\t".join(cells[ncols-1:])]
            rows.append(cells)
            buf = ""

    if buf:
        cells = buf.split("\t")
        if len(cells) == ncols:
            rows.append(cells)

    return pd.DataFrame(rows, columns=header, dtype=str)

def count_tokens_table(df: pd.DataFrame) -> int:
    parts = ["\t".join([str(c) for c in df.columns])]
    for _, row in df.iterrows():
        parts.append("\t".join([str(x) if x is not None else "" for x in row.tolist()]))
    text = "\n".join(parts)
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len([tok for tok in re.split(r"\s+", text) if tok])

# ✅ 세분화된 셀 카운트 버킷
cell_bins = {
    "<100": [],
    "100-199": [],
    "200-299": [],
    "300-399": [],
    "400-500": [],
    ">500": [],
}
token_bins = {"<2000": [], "2000-4000": [], ">4000": []}

tsv_files = []
for root, _, files in os.walk(TABLE_ROOT):
    for fn in files:
        if fn.endswith(".tsv"):
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, TABLE_ROOT)
            tsv_files.append((full, rel))

os.makedirs(OUT_DIR, exist_ok=True)
bad_files = []

for full_path, rel_path in tsv_files:
    try:
        df = read_tsv_by_tabcount(full_path)

        rows, cols = df.shape
        cell_count = rows * cols

        # 🔢 세분화된 분기
        if cell_count < 100:
            cell_bins["<100"].append(rel_path)
        elif 100 <= cell_count <= 199:
            cell_bins["100-199"].append(rel_path)
        elif 200 <= cell_count <= 299:
            cell_bins["200-299"].append(rel_path)
        elif 300 <= cell_count <= 399:
            cell_bins["300-399"].append(rel_path)
        elif 400 <= cell_count <= 500:
            cell_bins["400-500"].append(rel_path)
        else:
            cell_bins[">500"].append(rel_path)

        tok = count_tokens_table(df)
        if tok < 2000:
            token_bins["<2000"].append(rel_path)
        elif 2000 <= tok <= 4000:
            token_bins["2000-4000"].append(rel_path)
        else:
            token_bins[">4000"].append(rel_path)

    except Exception as e:
        bad_files.append((rel_path, str(e)))

def write_list(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for it in sorted(items):
            f.write(it + "\n")

# 저장: 셀
write_list(os.path.join(OUT_DIR, "cells_lt_100.txt"),  cell_bins["<100"])
write_list(os.path.join(OUT_DIR, "cells_100_199.txt"), cell_bins["100-199"])
write_list(os.path.join(OUT_DIR, "cells_200_299.txt"), cell_bins["200-299"])
write_list(os.path.join(OUT_DIR, "cells_300_399.txt"), cell_bins["300-399"])
write_list(os.path.join(OUT_DIR, "cells_400_500.txt"), cell_bins["400-500"])
write_list(os.path.join(OUT_DIR, "cells_gt_500.txt"),  cell_bins[">500"])

# 저장: 토큰
write_list(os.path.join(OUT_DIR, "tokens_lt_2000.txt"),   token_bins["<2000"])
write_list(os.path.join(OUT_DIR, "tokens_2000_4000.txt"), token_bins["2000-4000"])
write_list(os.path.join(OUT_DIR, "tokens_gt_4000.txt"),   token_bins[">4000"])

if bad_files:
    with open(os.path.join(OUT_DIR, "read_failures.txt"), "w", encoding="utf-8") as f:
        for rp, err in bad_files:
            f.write(f"{rp}\t{err}\n")

print("=== Size Buckets (by Cell Count) ===")
for k in ["<100","100-199","200-299","300-399","400-500",">500"]:
    print(f"{k:>8}: {len(cell_bins[k])}")

print("\n=== Size Buckets (by Token Count) ===")
print(f"<2000     : {len(token_bins['<2000'])}")
print(f"2000–4000 : {len(token_bins['2000-4000'])}")
print(f">4000     : {len(token_bins['>4000'])}")

if bad_files:
    print(f"\n[주의] 읽기 실패 파일 {len(bad_files)}개 → {os.path.join(OUT_DIR, 'read_failures.txt')}")
print(f"\n목록 저장 폴더: {OUT_DIR}")
