from .main import (
    generate_table_from_summary, 
    inf_evaluate,
    inf_feedback
)
from .TabEval import TableEvaluator, FeedbackTemplateManager

# 기존 호환성을 위한 별칭들
evaluate_generated_table = lambda original_table, generated_table, title="": TableEvaluator(use_cuda=True).evaluate_table_pair(original_table, generated_table, title)
FeedbackTemplates = FeedbackTemplateManager

__all__ = [
    'generate_table_from_summary',
    'inf_evaluate',
    'inf_feedback',
    'evaluate_generated_table', 
    'FeedbackTemplates',
    'TableEvaluator',
    'FeedbackTemplateManager'
]