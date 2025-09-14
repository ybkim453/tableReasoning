import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import euclidean_distances
import warnings
warnings.filterwarnings('ignore')

class ColumnSimilarityAnalyzer:
    def __init__(self, model_name='BAAI/bge-large-en'):
        self.model = SentenceTransformer(model_name)
        
    def calculate_column_similarities(self, column_values):
        # 모든 값을 문자열로 변환 (NaN도 그대로 유지)
        str_values = [str(val) for val in column_values]
        
        if len(str_values) < 2:
            return {}
        
        # 임베딩 생성
        embeddings = self.model.encode(str_values)
        
        # 유클리드 거리 행렬 계산 후 유사도로 변환
        distance_matrix = euclidean_distances(embeddings)
        # 거리를 유사도로 변환 (거리가 작을수록 유사도가 높음)
        similarity_matrix = 1 / (1 + distance_matrix)
        
        # 각 값의 유사도 총합 및 평균 계산
        similarities = {}
        for i, val in enumerate(str_values):
            # 자기 자신을 제외한 다른 값들과의 유사도
            other_similarities = [similarity_matrix[i][j] for j in range(len(str_values)) if i != j]
            
            similarities[i] = {
                'value': val,
                'similarity_sum': sum(other_similarities),
                'similarity_avg': np.mean(other_similarities) if other_similarities else 0,
                'similarity_scores': other_similarities
            }
        
        return similarities
    
    def analyze_table_columns(self, table_df, log_file_path):
        column_analyses = {}
        
        for col in table_df.columns:
            log_message = f"  컬럼 '{col}' 분석 중..."
            print(log_message)
            with open(log_file_path, 'a', encoding='utf-8') as log_f:
                log_f.write(log_message + '\n')
            
            similarities = self.calculate_column_similarities(table_df[col])
            
            if similarities:
                # 유사도 평균들의 분포 계산
                avg_similarities = [sim['similarity_avg'] for sim in similarities.values()]
                
                column_analyses[col] = {
                    'similarities': similarities,
                    'distribution_std': np.std(avg_similarities),
                    'min_similarity': min(avg_similarities),
                    'max_similarity': max(avg_similarities),
                    'mean_similarity': np.mean(avg_similarities)
                }
        
        return column_analyses
    
    def select_rows_for_subtable(self, table_df, column_analyses, log_file_path):
        selected_rows = []
        
        # 특이값 행 2개 선택
        extreme_columns = sorted(column_analyses.items(), 
                               key=lambda x: x[1]['distribution_std'], 
                               reverse=True)
        
        anomaly_count = 0
        used_columns = set()
        
        for col_name, col_info in extreme_columns:
            if anomaly_count >= 3:  # Fix: 특이값 행 개수 변경 시 여기 수정 (예: 3개면 >= 3)
                break
                
            if col_name in used_columns:
                continue
                
            # 낮은 유사도를 가진 값의 행 찾기
            min_sim_row = None
            min_sim_value = float('inf')
            
            for row_idx, sim_info in col_info['similarities'].items():
                if sim_info['similarity_avg'] < min_sim_value:
                    min_sim_value = sim_info['similarity_avg']
                    min_sim_row = row_idx
            
            if min_sim_row is not None and min_sim_row not in selected_rows:
                selected_rows.append(min_sim_row)
                used_columns.add(col_name)
                anomaly_count += 1
                log_message = f"  특이값 행 선택: 행 {min_sim_row} (컬럼 '{col_name}', 유사도: {min_sim_value:.4f})"
                print(log_message)
                with open(log_file_path, 'a', encoding='utf-8') as log_f:
                    log_f.write(log_message + '\n')
        
        # 정상값 행 2개 선택
        row_scores = {}
        
        for row_idx in range(len(table_df)):
            if row_idx in selected_rows:
                continue
                
            total_score = 0
            valid_columns = 0
            
            for col_name, col_info in column_analyses.items():
                if row_idx in col_info['similarities']:
                    total_score += col_info['similarities'][row_idx]['similarity_avg']
                    valid_columns += 1
            
            if valid_columns > 0:
                row_scores[row_idx] = total_score / valid_columns
        
        # 가장 높은 점수를 가진 행들 선택 (정확히 2개)
        normal_rows = sorted(row_scores.items(), key=lambda x: x[1], reverse=True)
        normal_count = 0
        
        for row_idx, score in normal_rows:
            if normal_count >= 3:  # Fix: 정상값 행 개수 변경 시 여기 수정 (예: 3개면 >= 3)
                break
            if row_idx not in selected_rows:
                selected_rows.append(row_idx)
                log_message = f"  정상값 행 선택: 행 {row_idx} (평균 유사도: {score:.4f})"
                print(log_message)
                with open(log_file_path, 'a', encoding='utf-8') as log_f:
                    log_f.write(log_message + '\n')
                normal_count += 1
        
        # 원본 테이블 순서대로 정렬하고 정확히 4개만 선택
        selected_rows = sorted(list(set(selected_rows)))[:6]  # FIX: 행 총 개수 변경 여기 수정 (예: 3X3이며 6개로 [:6])
        log_message = f"  최종 선택된 행 (원본 순서): {selected_rows} (총 {len(selected_rows)}개)"
        print(log_message)
        with open(log_file_path, 'a', encoding='utf-8') as log_f:
            log_f.write(log_message + '\n')
        
        return selected_rows
    

    
    def process_single_table(self, file_path, log_file_path):
        log_message = f"\n테이블 처리 중: {file_path}"
        print(log_message)
        with open(log_file_path, 'a', encoding='utf-8') as log_f:
            log_f.write(log_message + '\n')
        
        # 원본 파일의 모든 줄을 그대로 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        
        # 헤더와 구분선 (첫 2줄) - 원본 그대로 유지 (strip 하지 않음)
        header_line = all_lines[0].rstrip('\n\r')
        separator_line = all_lines[1].rstrip('\n\r')
        
        # 데이터 줄들 (3번째 줄부터)
        data_lines = [line.strip() for line in all_lines[2:]]
        
        # 분석을 위해서만 DataFrame 사용 (첫 번째 컬럼을 인덱스로 사용)
        temp_df = pd.read_csv(file_path, sep='\t', header=0, index_col=0, na_values=[''], keep_default_na=False)
        data_rows = temp_df.iloc[1:]  # 구분선 제외한 데이터만 (헤더는 이미 컬럼명)
        
        # 컬럼별 유사도 분석
        column_analyses = self.analyze_table_columns(data_rows, log_file_path)
        
        if not column_analyses:
            log_message = "  분석 가능한 컬럼이 없습니다."
            print(log_message)
            with open(log_file_path, 'a', encoding='utf-8') as log_f:
                log_f.write(log_message + '\n')
            # 상위 4개 데이터 줄 선택
            selected_data_lines = data_lines[:6]  # FIX: 행 총 개수 변경 여기 수정 (예: 3X3이며 6개로 [:6])
        else:
            # 서브테이블용 행 선택
            selected_rows = self.select_rows_for_subtable(data_rows, column_analyses, log_file_path)
            selected_data_lines = [data_lines[i] for i in selected_rows]
        
        # 결과: 헤더 + 구분선 + 선택된 4개 데이터 줄
        result_lines = [header_line, separator_line] + selected_data_lines
        log_message = f"  서브테이블 크기: {len(result_lines)}줄"
        print(log_message)
        with open(log_file_path, 'a', encoding='utf-8') as log_f:
            log_f.write(log_message + '\n')
        
        # ✅ result_lines + 선택된 인덱스 반환
        return result_lines, selected_rows
    

def main():
    # 입력/출력 경로 직접 지정
    input_path = "test_sub/sample.tsv"   # 처리할 원본 테이블 파일
    output_dir = "methodB_euclid_result_tt"  # 결과 저장할 디렉토리

    # 출력 폴더 생성
    os.makedirs(output_dir, exist_ok=True)

    # 분석기 초기화
    analyzer = ColumnSimilarityAnalyzer()

    # 로그 파일 경로
    log_file_path = os.path.join(output_dir, "log.txt")
    with open(log_file_path, 'w', encoding='utf-8') as log_f:
        log_f.write("=== 단일 테이블 처리 시작 ===\n")

    try:
        # 출력 파일명 구성
        filename = os.path.basename(input_path)
        output_filename = filename.replace('.tsv', '_sub.tsv')
        output_path = os.path.join(output_dir, output_filename)

        # 서브테이블 생성 (텍스트 줄들)
        result_lines = analyzer.process_single_table(input_path, log_file_path)

        # 결과 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in result_lines:
                f.write(line + '\n')

        log_message = f"  저장 완료: {output_path}"
        print(log_message)
        with open(log_file_path, 'a', encoding='utf-8') as log_f:
            log_f.write(log_message + '\n')

    except Exception as e:
        error_message = f"  오류 발생 ({input_path}): {str(e)}"
        print(error_message)
        with open(log_file_path, 'a', encoding='utf-8') as log_f:
            log_f.write(error_message + '\n')
    

if __name__ == "__main__":
    main()
