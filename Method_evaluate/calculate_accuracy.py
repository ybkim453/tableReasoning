import json
import re
import string
from collections import defaultdict

def normalize_answer(answer):
    if answer is None:
        return ""
    answer = str(answer).lower()
    answer = re.sub(r'\([^)]*\)', '', answer)  # 괄호와 그 안의 내용 제거
    answer = answer.translate(str.maketrans('', '', string.punctuation))  # 구두점 제거
    answer = re.sub(r'\s+', ' ', answer).strip()  # 공백 정리
    return answer

def is_correct_answer(ground_truth, model_answer):
    if ground_truth is None or model_answer is None:
        return False
    return normalize_answer(ground_truth) == normalize_answer(model_answer)

def calculate_accuracy():
    input_path = '/home/wooo519/tableReasoningFinal/MINE/MyMethod_evaluate/second/result_second.json'
    accuracy_path = '/home/wooo519/tableReasoningFinal/MINE/MyMethod_evaluate/second/result_second_with_accuracy.json'
    wrong_path = '/home/wooo519/tableReasoningFinal/MINE/MyMethod_evaluate/second/wrong_prediction.json'

    # 데이터 읽기
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    category_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    tabsize_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    total_stats = {'total': 0, 'correct': 0}
    wrong_predictions = []  # 오답 저장 리스트

    for item in data:
        answers = item.get('answer')
        model_answer = item.get('model_answer')
        category = item.get('category', 'Unknown')
        tabsize = item.get('tableSize', 'Unknown')

        # 전체
        total_stats['total'] += 1

        # 카테고리별
        category_stats[category]['total'] += 1

        # 테이블 사이즈별
        tabsize_stats[tabsize]['total'] += 1

        if is_correct_answer(answers, model_answer):
            total_stats['correct'] += 1
            category_stats[category]['correct'] += 1
            tabsize_stats[tabsize]['correct'] += 1
        else:
            wrong_predictions.append({
                "idx": item.get("idx"),
                "question": item.get("question"),
                "category": category,
                "tableSize": tabsize,
                "answer": answers,
                "model_answer": model_answer,
                "norm_answer": normalize_answer(answers),
                "norm_model_answer": normalize_answer(model_answer),
                "text": item.get("text", ""),
                "table_id": item.get("table_id")
            })
    
    # 전체 정확도
    overall_accuracy = (total_stats['correct'] / total_stats['total']) * 100
    detailed_results = {
        'overall_accuracy': {
            'correct': total_stats['correct'],
            'total': total_stats['total'],
            'percentage': round(overall_accuracy, 2)
        },
        'category_accuracy': {},
        'tabsize_accuracy': {}
    }

    # 카테고리별
    for category, stats in category_stats.items():
        accuracy = (stats['correct'] / stats['total']) * 100
        detailed_results['category_accuracy'][category] = {
            'correct': stats['correct'],
            'total': stats['total'],
            'percentage': round(accuracy, 2)
        }

    # tableSize별
    for tabsize, stats in tabsize_stats.items():
        accuracy = (stats['correct'] / stats['total']) * 100
        detailed_results['tabsize_accuracy'][tabsize] = {
            'correct': stats['correct'],
            'total': stats['total'],
            'percentage': round(accuracy, 2)
        }

    # 결과 저장
    with open(accuracy_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    with open(wrong_path, 'w', encoding='utf-8') as f:
        json.dump(wrong_predictions, f, ensure_ascii=False, indent=2)

    # 출력
    print("=" * 60)
    print("모델 정답률 분석 결과")
    print("=" * 60)
    print(f"\n전체 정답률: {total_stats['correct']}/{total_stats['total']} = {overall_accuracy:.2f}%")

    print(f"\n카테고리별 정답률:")
    for category, stats in detailed_results['category_accuracy'].items():
        print(f"- {category}: {stats['correct']}/{stats['total']} = {stats['percentage']:.2f}%")

    print(f"\n테이블 사이즈별 정답률:")
    for tabsize, stats in detailed_results['tabsize_accuracy'].items():
        print(f"- {tabsize}: {stats['correct']}/{stats['total']} = {stats['percentage']:.2f}%")

    print(f"\n상세 결과가 {accuracy_path} 에 저장되었습니다.")
    print(f"오답 사례가 {wrong_path} 에 저장되었습니다.")
    print("\n오답 예시 (최초 3개):")
    for wp in wrong_predictions[:3]:
        print(f"- idx {wp['idx']} | 정답: {wp['answer']} | 모델답: {wp['model_answer']} | tableSize={wp['tableSize']}")

if __name__ == "__main__":
    calculate_accuracy()
