# Ollama 버전 테이블 파이프라인 (모듈화)
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from utils.few_shot_dp import few_model_1, few_user_1
from tabHint_gen import SubtableGenerator, TableHintGenerator
from tabHint_eval import inf_evaluate, inf_feedback, FeedbackTemplateManager

# Load environment variables from .env file
load_dotenv()

# Ollama 클라이언트 설정
ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# 모듈 초기화
subtable_gen = SubtableGenerator()
tabhint_gen = TableHintGenerator(ollama_client)

def get_resume_idx(result_path):
    """resume idx 가져오기"""
    if not os.path.exists(result_path):
        return 0
    with open(result_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if not lines:
            return 0
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
                return last["idx"] + 1
            except json.JSONDecodeError:
                continue
        return 0

def run_all(dataset_path: str,
            model_name: str = "gemma3:27b",
            log_dir: str = "output",
            result_path: str = "output/result.jsonl",
            enable_table_generation: bool = True,
            total_threshold: float = 0.65,  # 전체 점수 기준값
            em_threshold: float = 0.7,      # 개별 피드백용
            chrf_threshold: float = 0.6,    # 개별 피드백용
            bert_threshold: float = 0.65,   # 개별 피드백용
            max_iterations: int = 3):
    
    os.makedirs(log_dir, exist_ok=True)

    # Dataset 로드
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"Dataset loaded, size={len(dataset)}")

    base_table_dir = "../wtq"
    start_idx = get_resume_idx(result_path)
    print(f"Resume from idx={start_idx}")

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

            # 1. Subtable 생성
            subtable_result = subtable_gen.generate_subtable(file_path, log_dir, global_idx)
            if not subtable_result['success']:
                print(f"❌ Subtable generation failed: {subtable_result['error']}")
                continue
            
            preview_table = subtable_result['preview_table']
            full_table = subtable_result['full_table']

            # 2. Table Summary 생성
            table_summary = tabhint_gen.generate_with_cache(
                preview_table, model_name, log_dir, table_id
            )

            # 3. Table Generation + Evaluation + Feedback (선택적)
            table_gen_log = {}
            if enable_table_generation:
                print(f"🔧 Starting table generation for {table_id}")
                
                current_summary = table_summary
                iterations = []
                iteration = 0
                total_score = 0  # 초기화
                
                # While 루프: total_score 기준 + max_iterations 안전장치
                while total_score < total_threshold and iteration < max_iterations:
                    iteration += 1
                    print(f"  🔄 Iteration {iteration}")
                    
                    # main.py 호출: table summary → generation → evaluation
                    result = inf_evaluate(
                        table_summary=current_summary,
                        subtable=preview_table,
                        title=title
                    )
                    
                    if not result['success']:
                        print(f"    ❌ Evaluation failed: {result['error']}")
                        break
                    
                    scores = result['scores']
                    em_score = scores['em_f1']
                    chrf_score = scores['chrf_f1']
                    bert_score = scores['bert_f1']
                    
                    # run_table_ollama.py에서 total_score 계산 (0.2*EM + 0.4*chrF + 0.4*BERT)
                    total_score = (em_score * 0.2) + (chrf_score * 0.4) + (bert_score * 0.4)
                    print(f"    📊 Scores: EM={em_score:.3f}, chrF={chrf_score:.3f}, BERT={bert_score:.3f}, Total={total_score:.3f}")
                    
                    # 전체 점수 기준으로 통과 여부 확인 (run_table_ollama.py에서)
                    passed = bool(total_score >= total_threshold)
                    
                    iteration_data = {
                        'iteration': iteration,
                        'generated_table': result['generated_table'],
                        'scores': scores,
                        'passed': passed,
                        'current_summary': current_summary
                    }
                    
                    iterations.append(iteration_data)
                    
                    if passed:
                        print(f"    ✅ Total score threshold passed! ({total_score:.3f} >= {total_threshold})")
                    else:
                        print(f"    ⚠️ Total score below threshold ({total_score:.3f} < {total_threshold})")
                        # 개별 메트릭 기준으로 피드백 생성 (모듈에서)
                        feedback = inf_feedback(scores, em_threshold, chrf_threshold, bert_threshold)
                        if feedback:
                            # 원래 프롬프트 + 피드백으로 새로운 table summary 생성
                            feedback_prompt = f"""
{tabhint_gen.prompt_template.format(table_markdown=preview_table)}

**[개선 요청 - Iteration {iteration}]**
{feedback}

위 피드백을 반영하여 더 정확하고 적절한 테이블 분석을 제공해주세요.
"""
                            
                            response = ollama_client.chat.completions.create(
                                model=model_name,
                                messages=[{"role": "user", "content": feedback_prompt}],
                                temperature=0.1
                            )
                            current_summary = response.choices[0].message.content.strip()
                            
                            iteration_data['feedback'] = feedback
                            iteration_data['improved_summary'] = current_summary
                            print(f"    🔄 Table summary regenerated with feedback")
                        else:
                            print("    ⚠️ No feedback generated but scores not passed")
                            break
                
                # 결과 정리
                success = len(iterations) > 0 and iterations[-1].get('passed', False)
                table_gen_log = {
                    'enabled': True,
                    'success': success,
                    'total_iterations': len(iterations),
                    'iterations': iterations
                }
                
                # Table generation + evaluation 로그를 summary_plus_eval에 추가
                summary_eval_dir = os.path.join(log_dir, "summary_plus_eval")
                os.makedirs(summary_eval_dir, exist_ok=True)
                summary_eval_path = os.path.join(summary_eval_dir, f"{table_id.replace('/', '_')}.txt")
                
                # 각 iteration별로 로그 작성
                with open(summary_eval_path, "w", encoding="utf-8") as f:  # overwrite mode
                    for i, iteration in enumerate(table_gen_log.get('iterations', [])):
                        # Table Summary 섹션
                        f.write(f"=================== Table Summary ===================\n")
                        f.write(f"{iteration['current_summary']}\n\n")
                        
                        # Table Generation 섹션
                        f.write(f"=================== Table Generation ===================\n")
                        f.write(f"{iteration.get('generated_table', 'N/A')}\n\n")
                        
                        # Summary Eval 섹션
                        f.write(f"=================== Summary Eval ===================\n")
                        scores = iteration['scores']
                        em_score = scores['em_f1']
                        chrf_score = scores['chrf_f1']
                        bert_score = scores['bert_f1']
                        total_score = (em_score * 0.2) + (chrf_score * 0.4) + (bert_score * 0.4)
                        
                        f.write(f"Evaluation Scores:\n")
                        f.write(f"- EM: {em_score:.3f}\n")
                        f.write(f"- chrF: {chrf_score:.3f}\n")
                        f.write(f"- BERT: {bert_score:.3f}\n")
                        f.write(f"- Total: {total_score:.3f}\n")
                        f.write(f"- Passed: {iteration.get('passed', False)}\n")
                        
                        if 'feedback' in iteration:
                            f.write(f"\nFeedback for Next Iteration:\n")
                            f.write(f"{iteration['feedback']}\n")
                        
                        f.write(f"\n")  # iteration 간 구분
                
                print(f"📝 Table generation completed: {table_gen_log['success']}")
            else:
                table_gen_log = {'enabled': False}

            # 4. QA 처리
            for q_idx, (question, answer, qid) in enumerate(
                zip(d["questions"], d["answers"], d["ids"])
            ):
                if global_idx < start_idx:
                    global_idx += 1
                    continue

                # QA 시스템 프롬프트
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

                # QA 실행 (5번 샘플링)
                step_outputs = []
                for _ in range(5):
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

                # Reasoning 로그 저장
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

                # JSONL 결과 저장
                res = {
                    "idx": global_idx,
                    "answer": answer,
                    "text": step_outputs,
                    "question_id": qid,
                    "table_id": table_id,
                    "title": title,
                    "question": question,
                    "table_generation": table_gen_log,
                }
                fout.write(json.dumps(res) + "\n")

                print(f"✅ Finished idx={global_idx}, table_id={table_id}")
                global_idx += 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Table Pipeline with Optional Table Generation (Ollama)")
    parser.add_argument('--dataset_path', default="./data/wtq.json", help="Dataset path")
    parser.add_argument('--model_name', default="gemma3:27b", help="Model name")
    parser.add_argument('--log_dir', default="output", help="Log directory")
    parser.add_argument('--result_path', default="output/result.jsonl", help="Result path")
    parser.add_argument('--disable_table_generation', action='store_true', help="Disable table generation")
    parser.add_argument('--enable_table_generation', action='store_true', help="Enable table generation (default: True)")
    parser.add_argument('--total_threshold', type=float, default=0.65, help="Total score threshold for passing")
    parser.add_argument('--em_threshold', type=float, default=0.7, help="EM threshold for feedback")
    parser.add_argument('--chrf_threshold', type=float, default=0.6, help="chrF threshold for feedback")
    parser.add_argument('--bert_threshold', type=float, default=0.65, help="BERT threshold for feedback")
    parser.add_argument('--max_iterations', type=int, default=3, help="Max refinement iterations")
    
    args = parser.parse_args()
    
    # Table generation 기본 활성화, --disable_table_generation으로 비활성화 가능
    enable_table_gen = not args.disable_table_generation
    if args.enable_table_generation:  # 명시적으로 활성화 요청시
        enable_table_gen = True
    
    run_all(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        log_dir=args.log_dir,
        result_path=args.result_path,
        enable_table_generation=enable_table_gen,
        total_threshold=args.total_threshold,
        em_threshold=args.em_threshold,
        chrf_threshold=args.chrf_threshold,
        bert_threshold=args.bert_threshold,
        max_iterations=args.max_iterations
    )