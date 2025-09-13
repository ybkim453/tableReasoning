# 최종 전체 테스트용 (resume 지원 버전)
import os
import csv
import json
from io import StringIO

from openai import OpenAI
import pandas as pd
from subtable_extractor_euclid import ColumnSimilarityAnalyzer
from utils.few_shot_dp import few_model_1, few_user_1

analyzer = ColumnSimilarityAnalyzer()

ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# === GPT로 테이블 요약을 요청하는 함수 ===
def call_gpt_table_summary(table_markdown: str, model: str = "gemma3:27b") -> str:
    prompt = f"""
You are a strict table analysis assistant.
Here is a partial preview of a table in markdown format:

{table_markdown}

Task: generate a structured HINT PACK with EXACTLY six sections in this order. 
Each section must not just list values but also provide concise explanatory sentences interpreting the evidence.
---

### 1) Column-wise Information
- For **every column**, follow this exact format:
    - **ColumnName**:
        - Data type (numeric, categorical, text, date, etc.).
        - If mixed types occur (numeric + placeholder, percentage + text), explicitly describe the mixture with ≥3 examples.
        - At least 4 explicit example values from the preview rows.  
        - Role of the column in isolation (identifier, descriptor, score field, etc.).
        - Explicitly note uncertain/ambiguous markers ("?", "N/A", "—") that may require schema-linking.

### 2) Special / Atypical Rows
- Identify rows that must be treated specially (Totals, Averages, World rows, placeholders like "?", "N/A", "—").
- Quote exact values and row contexts.
- Explain clearly why these rows should be excluded or specially handled in reasoning.

### 3) Derived / Relational Hints
- **Derived**: show explicit formulas and demonstrate with at least 2 rows.
- **Relational**: deterministic patterns (e.g., Position="—" ↔ Notes="DNF"), quoting paired evidence.
- If none: state "none observed — inconsistent or incompatible forms".

### 4) Sorting / Structural Hints
- Explicitly check all candidate columns (Rank, Year, etc.) for ordering.
- Provide ≥4 consecutive values as evidence.
- If table is partial, note possible gaps but still describe the apparent order.
- If multiple columns impose order, specify primary vs secondary.
- If unclear, write "unclear".

### 5) Preprocessing Artifacts
- (A) Syntactic artifacts: missing values (nan, None, etc.), duplicated rows, parsing leftovers (quote examples).
- (B) Mixed literal expressions: cases where semantic types are combined (e.g., "59% (2013)").
- (C) Semantic normalization — Units & Scales: inconsistencies (%, $, ranges). Propose plain normalization targets (e.g., Price → USD, Rate → [0–1], Distance → meters).

### 6) Abbreviations / Acronyms & Aliases
- List abbreviations or shortened headers (e.g., "ASL", "N/A") and attempt expansions.
- Explicitly treat values like "N/A" as abbreviations (Not Applicable).
- Include aliases/synonyms useful for schema-linking (e.g., Team ↔ Club, City ↔ Location).
- If unclear, write "unclear".

### STRICT RULES:
- Use only the six sections and headings in this exact order.
- Every claim must include direct evidence from the table.
- Do not estimate unseen statistics or frequencies.
- Do not include special rows in arithmetic unless explicitly justified.
- If uncertain, always write "unclear".

### Output template:

Column-wise Information:
...

Special / Atypical Rows:
...

Derived / Relational Hints:
...

Sorting / Structural Hints:
...

Preprocessing Artifacts:
...

Abbreviations / Acronyms:
...
"""
    response = ollama_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    summary = response.choices[0].message.content.strip()
    return summary

# === resume idx 가져오기 ===
def get_resume_idx(result_path):
    if not os.path.exists(result_path):
        return 0
    with open(result_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if not lines:
            return 0
        # 뒤에서부터 돌면서 제대로 파싱되는 JSON 찾기
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
                return last["idx"] + 1
            except json.JSONDecodeError:
                continue
        # 끝까지 못 찾으면 처음부터 시작
        return 0

def run_all(dataset_path: str,
            model_name: str = "gemma3:27b",
            log_dir: str = "output_final_second_log",
            result_path: str = "result.jsonl"):
    
    os.makedirs(log_dir, exist_ok=True)

    # === dataset 로드 ===
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"=== [DEBUG] Dataset loaded, size={len(dataset)} ===")

    base_table_dir = "../wtq"

    # resume idx 불러오기
    start_idx = get_resume_idx(result_path)
    print(f"=== [DEBUG] Resume from idx={start_idx} ===")

    # append 모드
    fout_mode = "a" if start_idx > 0 else "w"

    global_idx = 0
    with open(result_path, fout_mode, encoding="utf-8") as fout:
        for d in dataset:
            table_id = d["table_id"]
            title = d.get("title", table_id)

            file_path = os.path.join(base_table_dir, table_id)
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}, skipping")
                continue

            # === ColumnSimilarity 기반 preview 생성 ===
            subtable_dir = os.path.join(log_dir, "subtable_logs")
            os.makedirs(subtable_dir, exist_ok=True)

            subtable_path = os.path.join(subtable_dir, f"{global_idx}_log.txt")
            result_lines, selected_rows = analyzer.process_single_table(file_path, subtable_path)

            selected_rows = [r + 1 for r in selected_rows]
            
            # Sub-table DataFrame
            csv_content = "\n".join([result_lines[0]] + result_lines[2:])
            preview_df = pd.read_csv(
                StringIO(csv_content),
                sep="\t",
                quoting=csv.QUOTE_NONE,
                on_bad_lines="skip"
            )

            if "Unnamed: 0" in preview_df.columns:
                preview_df = preview_df.drop(columns=["Unnamed: 0"])
            preview_df.insert(0, " ", selected_rows)
            preview_table = preview_df.to_markdown(index=False)

            full_df = pd.read_csv(
                file_path,
                sep="\t",
                quoting=csv.QUOTE_NONE,
                on_bad_lines="skip"
            )
            full_table = full_df.to_markdown()

            # === summary 테이블 생성 ===
            summary_dir = os.path.join(log_dir, "summary_logs")
            os.makedirs(summary_dir, exist_ok=True)
            summary_path = os.path.join(summary_dir, f"{table_id.replace('/', '_')}.txt")
            if os.path.exists(summary_path):
                with open(summary_path, "r", encoding="utf-8") as f:
                    table_summary = f.read()
                print(f"⚡ Using cached summary for {table_id}")
            else:
                table_summary = call_gpt_table_summary(preview_table, model=model_name)
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write("=================== Table Summary ===================\n")
                    f.write(table_summary + "\n")


            for q_idx, (question, answer, qid) in enumerate(
                zip(d["questions"], d["answers"], d["ids"])
            ):
                # === resume 체크 ===
                if global_idx < start_idx:
                    global_idx += 1
                    continue

                # === reasoning system prompt ===
                system_prompt = """
                The provided summary is the primary structure for interpreting the Full Table.  
                Always start by carefully examining the summary to identify which columns, data types, and special cases are relevant to the question.  
                Then, use the full table to confirm and extract the exact values based on the hints from the summary.  
                Your reasoning must explicitly combine evidence from both the summary and the full table before giving the final answer.  
                Final output must follow the format:
                Final Answer: AnswerName1, AnswerName2...
                """

                user_prompt = f"""
                Here is a summary of the table :
                {table_summary}

                Here is the full table regarding "{title}". This is the result of `print(df.to_markdown())` :
                {full_table}

                Question :
                {question}
                """.strip()

                step_outputs = []
                num_samples = 5

                for _ in range(num_samples):
                    response = ollama_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": few_user_1},
                            {"role": "assistant", "content": few_model_1},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.3,
                    )
                    step_outputs.append(response.choices[0].message.content.strip())

                # === reasoning 로그 저장 ===
                reasoning_dir = os.path.join(log_dir, "reasoning_logs")
                os.makedirs(reasoning_dir, exist_ok=True)
                reasoning_path = os.path.join(reasoning_dir, f"{global_idx}_{qid}.txt")
                with open(reasoning_path, "w", encoding="utf-8") as f:
                    f.write("=================== Title ===================\n")
                    f.write("title : " + title + "  & table id : " + table_id + "\n\n")
                    f.write("=================== Table ===================\n")
                    f.write(full_table + "\n\n")
                    f.write("=================== Question ===================\n")
                    f.write(question + "\n\n")
                    for i, out in enumerate(step_outputs, 1):
                        f.write(f"=================== Reasoning {i} ===================\n")
                        f.write(out + "\n\n")
                    f.write("=================== Answer ===================\n")
                    f.write(",".join(answer) if isinstance(answer, list) else str(answer))
                    f.write("\n")

                # === jsonl 저장 ===
                res = {
                    "idx": global_idx,
                    "answer": answer,
                    "text": step_outputs,
                    "question_id": qid,
                    "table_id": table_id,
                    "title": title,
                    "question": question,
                }
                fout.write(json.dumps(res) + "\n")

                print(f"✅ Finished idx={global_idx}, table_id={table_id}")
                global_idx += 1


if __name__ == "__main__":
    run_all(
        dataset_path="./data/wtq.json",
        model_name="gemma3:27b",
        log_dir="output",
        result_path="output/result.jsonl"
    )
