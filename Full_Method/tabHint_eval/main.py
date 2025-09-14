"""
Table Generation & Evaluation 메인 모듈
table summary → table generation → evaluation → feedback 반환
"""

import os
import sys
from pathlib import Path
import tempfile
from typing import Dict, Tuple, Optional

from .TabGen import CheckTableGenerator
from .TabEval import TableEvaluator, FeedbackTemplateManager

# Table Generation
def generate_table_from_summary(table_summary: str, title: str = "") -> str:
    try:
        generator = CheckTableGenerator()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            content = f"""=================== Title ===================
{title}

=================== Table Summary ===================
{table_summary}
"""
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            with tempfile.TemporaryDirectory() as temp_output_dir:
                output_path = generator.process_file(str(temp_file_path), str(temp_output_dir))
                
                if output_path:
                    temp_filename = Path(temp_file_path).stem
                    generated_table_path = Path(temp_output_dir) / f"{temp_filename}_step3_generated_table.md"
                    
                    if generated_table_path.exists():
                        with open(generated_table_path, 'r', encoding='utf-8') as f:
                            return f.read()
                    else:
                        return f"Error: Generated table file not found at {generated_table_path}"
                else:
                    return "Error: Table generation failed - no output path returned"
                    
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        return f"Error: {str(e)}"


def inf_evaluate(table_summary: str, subtable: str, title: str = "") -> Dict:
    try:
        # 1. table generation
        generated_table = generate_table_from_summary(table_summary, title)
        
        if generated_table.startswith("Error:"):
            return {
                'success': False,
                'error': generated_table,
                'scores': {},
                'generated_table': '',
                'detailed_metrics': {}
            }
        
        # 2. evaluation
        evaluator = TableEvaluator(use_cuda=True) 
        eval_result = evaluator.evaluate_table_pair(subtable, generated_table, title)
        
        if not eval_result['success']:
            return {
                'success': False,
                'error': eval_result['error'],
                'scores': {},
                'generated_table': generated_table,
                'detailed_metrics': {}
            }
        
        return {
            'success': True,
            'scores': eval_result['scores'],
            'generated_table': generated_table,
            'detailed_metrics': eval_result['detailed_metrics']
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Evaluation failed: {str(e)}',
            'scores': {},
            'generated_table': '',
            'detailed_metrics': {}
        }


def inf_feedback(scores: Dict, em_threshold: float = 0.7, chrf_threshold: float = 0.6, 
                bert_threshold: float = 0.65) -> str:
    try:
        em_score = scores.get('em_f1', 0)
        chrf_score = scores.get('chrf_f1', 0) 
        bert_score = scores.get('bert_f1', 0)
        
        feedback_manager = FeedbackTemplateManager(
            em_threshold=em_threshold,
            chrf_threshold=chrf_threshold,
            bert_threshold=bert_threshold
        )
        
        return feedback_manager.generate_feedback_prompt(em_score, chrf_score, bert_score)
        
    except Exception as e:
        return f"Feedback generation failed: {str(e)}"