"""
Table Hint Generation 모듈
테이블 요약(hint) 생성 담당
"""

import os
from pathlib import Path
from openai import OpenAI


class TableHintGenerator:
    """테이블 힌트(요약) 생성 클래스"""
    
    def __init__(self, client: OpenAI):
        """
        Args:
            client: OpenAI 클라이언트 (ollama 또는 openai)
        """
        self.client = client
        self.prompt_template = self._load_prompt_template()
    
    def _load_prompt_template(self) -> str:
        """프롬프트 템플릿 로드"""
        prompt_path = Path(__file__).parent / "tabhint_prompt.txt"
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        else:
            return "Error: tabhint_prompt.txt not found"
    
    def generate_table_summary(self, table_markdown: str, model: str, 
                             cache_path: str = None) -> str:
        """
        테이블 요약 생성
        
        Args:
            table_markdown: 테이블 마크다운 텍스트
            model: 사용할 모델명
            cache_path: 캐시 파일 경로 (있으면 캐시 사용)
            
        Returns:
            생성된 테이블 요약
        """
        # 캐시 확인
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_summary = f.read()
                if cached_summary.strip():
                    return cached_summary
        
        try:
            # 프롬프트 생성
            prompt = self.prompt_template.format(table_markdown=table_markdown)
            
            # 모델 호출
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            summary = response.choices[0].message.content.strip()
            
            # 캐시 저장
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write("=================== Table Summary ===================\n")
                    f.write(summary + "\n")
            
            return summary
            
        except Exception as e:
            return f"Error generating table summary: {str(e)}"
    
    def generate_with_cache(self, table_markdown: str, model: str, 
                          log_dir: str, table_id: str) -> str:
        """
        캐시를 활용한 테이블 요약 생성
        
        Args:
            table_markdown: 테이블 마크다운
            model: 모델명
            log_dir: 로그 디렉토리
            table_id: 테이블 ID
            
        Returns:
            테이블 요약
        """
        # 캐시 경로 설정
        summary_dir = os.path.join(log_dir, "summary_plus_eval")
        cache_path = os.path.join(summary_dir, f"{table_id.replace('/', '_')}.txt")
        
        # 캐시 확인
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                print(f"⚡ Using cached summary for {table_id}")
                return f.read()
        
        # 새로 생성
        summary = self.generate_table_summary(table_markdown, model, cache_path)
        return summary
