import json

def jsonl_to_json(jsonl_path, json_path):
    data = []
    decoder = json.JSONDecoder()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx = 0
            while idx < len(line):
                obj, pos = decoder.raw_decode(line, idx)
                data.append(obj)
                idx = pos

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 실행
jsonl_to_json(
    "/home/wooo519/tableReasoningFinal/MINE/MyMethod_evaluate/second/result.jsonl",
    "/home/wooo519/tableReasoningFinal/MINE/MyMethod_evaluate/second/result.json"
)
