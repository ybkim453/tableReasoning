# Category, TabSize & 모델 정답 추출
import json
import os
import re

# 입력 파일 (원본 결과 JSON)
input_result_file = "/home/wooo519/tableReasoningFinal/MINE/MyMethod_evaluate/second/result.json"
# category 매핑 파일
input_category_file = "/home/wooo519/tableReasoningFinal/MINE/MyMethod_test_dataset/wtq_question_category.json"
# 출력 파일
output_file = "/home/wooo519/tableReasoningFinal/MINE/MyMethod_evaluate/second/result_second.json"

# --- 1) category 매핑 로드 ---
with open(input_category_file, "r", encoding="utf-8") as f:
    data_b = json.load(f)

category_by_idx = {item["idx"]: item["category"] for item in data_b}

# --- 2) tableSize 매핑 로드 ---
base_dir = "/home/wooo519/tableReasoningFinal/MINE/MyMethod_evaluate/alter_tab_size/tabsize_WTQ_test/tokens"
txt_files = {
    "less_2000": os.path.join(base_dir, "tokens_less_2000.txt"),
    "2000_4000": os.path.join(base_dir, "tokens_2000_4000.txt"),
    "over_4000": os.path.join(base_dir, "tokens_over_4000.txt"),
}

table_groups = {}
for name, path in txt_files.items():
    with open(path, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    table_groups[name] = set(ids)

print("✅ Loaded table groups:")
for k, v in table_groups.items():
    print(f"{k}: {len(v)} entries")

# --- 3) 원본 JSON 불러오기 ---
with open(input_result_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# --- 4) model_answer 추출용 패턴 ---
pattern = re.compile(r"Final Answer:\s*(.*)", re.IGNORECASE)

# --- 5) 각 항목 처리 ---
for obj in data:
    idx = obj.get("idx")
    table_id = obj.get("table_id")

    # (a) category 추가
    obj["category"] = category_by_idx.get(idx, "Unknown")

    # (b) tableSize 추가 (tokens_ 접두사, .txt 제거 → 깔끔하게)
    table_size = "unknown"
    for name, id_set in table_groups.items():
        if table_id in id_set:
            table_size = name
            break
    obj["tableSize"] = table_size

    # (c) model_answer 추출
    text = obj.get("text", "")
    match = pattern.search(text)
    if match:
        final_answer = match.group(1).strip()
        obj["model_answer"] = final_answer
        obj["text"] = pattern.sub("", text).strip()
    else:
        obj["model_answer"] = None

# --- 6) 저장 ---
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"🎉 Done! 결과는 {output_file} 에 저장되었습니다.")
