# GPT version table pipeline
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from utils.few_shot_dp import few_model_1, few_user_1
from tabHint_gen import SubtableGenerator, TableHintGenerator
from tabHint_eval import inf_evaluate, inf_feedback, FeedbackTemplateManager

# Load environment variables from .env file
load_dotenv()

openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

subtable_gen = SubtableGenerator()
tabhint_gen = TableHintGenerator(openai_client)

def get_resume_idx(result_path):
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
            model_name: str = "gpt-3.5-turbo",
            log_dir: str = "output_gpt",
            result_path: str = "output_gpt/result.jsonl",
            enable_table_generation: bool = True,
            total_threshold: float = 0.65,  # total score threshold
            em_threshold: float = 0.6,      # individual feedback
            chrf_threshold: float = 0.6,    # individual feedback
            bert_threshold: float = 0.65,   # individual feedback
            max_iterations: int = 3,
            gpu_id: int = None):            # GPU 선택 (None=자동선택)
    
    os.makedirs(log_dir, exist_ok=True)

    # load dataset
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
                print(f"File not found: {file_path}, skipping")
                continue

            # 1. Subtable generation
            subtable_result = subtable_gen.generate_subtable(file_path, log_dir, global_idx)
            if not subtable_result['success']:
                print(f"Subtable generation failed: {subtable_result['error']}")
                continue
            
            preview_table = subtable_result['preview_table']
            full_table = subtable_result['full_table']

            # 2. Table Summary generation
            table_summary = tabhint_gen.generate_with_cache(
                preview_table, model_name, log_dir, table_id
            )

            # 3. Table Generation + Evaluation + Feedback
            table_gen_log = {}
            if enable_table_generation:
                print(f"Starting table generation for {table_id}")
                
                current_summary = table_summary
                iterations = []
                iteration = 0
                
                result = inf_evaluate(
                    table_summary=current_summary,
                    subtable=preview_table,
                    title=title
                )
                
                if not result['success']:
                    print(f"Initial evaluation failed: {result['error']}")
                    table_gen_log = {'enabled': True, 'success': False, 'error': result['error']}
                else:
                    scores = result['scores']
                    em_score = scores['em_f1']
                    chrf_score = scores['chrf_f1'] 
                    bert_score = scores['bert_f1']
                    total_score = (em_score * 0.2) + (chrf_score * 0.4) + (bert_score * 0.4)
                    
                    print(f"Initial Scores: EM={em_score:.3f}, chrF={chrf_score:.3f}, BERT={bert_score:.3f}, Total={total_score:.3f}")
                    
                    initial_data = {
                        'iteration': 0,
                        'generated_table': result['generated_table'],
                        'scores': scores,
                        'passed': bool(total_score >= total_threshold),
                        'current_summary': current_summary
                    }
                    iterations.append(initial_data)
                    
                    # total_score < threshold인 경우에만 개선 루프 실행
                    while total_score < total_threshold and iteration < max_iterations:
                        iteration += 1
                        print(f"Refinement Iteration {iteration}")

                        feedback = inf_feedback(scores, em_threshold, chrf_threshold, bert_threshold)
                        if not feedback:
                            print("No feedback generated")
                            break

                        feedback_prompt = f"""
{tabhint_gen.prompt_template.format(table_markdown=preview_table)}

**[Improvement request - Iteration {iteration}]**
{feedback}

Please provide a more accurate and appropriate table analysis by reflecting the above feedback.
"""
                        
                        response = openai_client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": feedback_prompt}],
                            temperature=0.1
                        )
                        current_summary = response.choices[0].message.content.strip()

                        result = inf_evaluate(
                            table_summary=current_summary,
                            subtable=preview_table,
                            title=title
                        )
                        
                        if not result['success']:
                            print(f"Re-evaluation failed: {result['error']}")
                            break
                        
                        scores = result['scores']
                        em_score = scores['em_f1']
                        chrf_score = scores['chrf_f1']
                        bert_score = scores['bert_f1'] 
                        total_score = (em_score * 0.2) + (chrf_score * 0.4) + (bert_score * 0.4)
                        
                        print(f"Iteration {iteration} Scores: EM={em_score:.3f}, chrF={chrf_score:.3f}, BERT={bert_score:.3f}, Total={total_score:.3f}")

                        iteration_data = {
                            'iteration': iteration,
                            'generated_table': result['generated_table'],
                            'scores': scores,
                            'passed': bool(total_score >= total_threshold),
                            'current_summary': current_summary,
                            'feedback': feedback,
                            'improved_summary': current_summary
                        }
                        iterations.append(iteration_data)
                    
                    success = bool(total_score >= total_threshold)
                
                # table_gen_log
                table_gen_log = {
                    'enabled': True,
                    'success': success,
                    'total_iterations': len(iterations),
                    'iterations': iterations
                }
                
                # Table generation + evaluation log to summary_plus_eval
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
                
                print(f"Table generation completed: {table_gen_log['success']}")
            else:
                table_gen_log = {'enabled': False}

            # 4. QA processing
            for q_idx, (question, answer, qid) in enumerate(
                zip(d["questions"], d["answers"], d["ids"])
            ):
                if global_idx < start_idx:
                    global_idx += 1
                    continue

                # QA system prompt
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

                # QA execution (5 sampling)
                step_outputs = []
                for _ in range(5):
                    response = openai_client.chat.completions.create(
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

                # Reasoning log saving
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

                # JSONL result saving
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

                print(f"Finished idx={global_idx}, table_id={table_id}")
                global_idx += 1


if __name__ == "__main__":
    import argparse
    
    # check OPENAI_API_KEY environment variable
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        exit(1)
    
    parser = argparse.ArgumentParser(description="Table Pipeline with Optional Table Generation (GPT)")
    parser.add_argument('--dataset_path', default="./data/wtq.json", help="Dataset path")
    parser.add_argument('--model_name', default="gpt-3.5-turbo", help="Model name")
    parser.add_argument('--log_dir', default="output_gpt", help="Log directory")
    parser.add_argument('--result_path', default="output_gpt/result.jsonl", help="Result path")
    parser.add_argument('--disable_table_generation', action='store_true', help="Disable table generation")
    parser.add_argument('--enable_table_generation', action='store_true', help="Enable table generation (default: True)")
    parser.add_argument('--total_threshold', type=float, default=0.65, help="Total score threshold for passing")
    parser.add_argument('--em_threshold', type=float, default=0.7, help="EM threshold for feedback")
    parser.add_argument('--chrf_threshold', type=float, default=0.6, help="chrF threshold for feedback")
    parser.add_argument('--bert_threshold', type=float, default=0.65, help="BERT threshold for feedback")
    parser.add_argument('--max_iterations', type=int, default=3, help="Max refinement iterations")
    parser.add_argument('--gpu_id', type=int, default=None, help="GPU ID to use (None=auto select)")
    
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
        max_iterations=args.max_iterations,
        gpu_id=args.gpu_id
    )