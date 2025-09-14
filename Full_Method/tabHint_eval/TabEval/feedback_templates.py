# Feedback template manager
# EM, chrF, BERTScore based specific feedback prompt

from typing import Dict
from pathlib import Path


class FeedbackTemplateManager:

    def __init__(self, 
                 em_threshold: float = 0.7,
                 chrf_threshold: float = 0.6, 
                 bert_threshold: float = 0.65):
        self.em_threshold = em_threshold
        self.chrf_threshold = chrf_threshold
        self.bert_threshold = bert_threshold
        self.prompts_dir = Path(__file__).parent / "prompts"
    
    def _load_feedback_template(self, filename: str) -> str:
        template_path = self.prompts_dir / filename
        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        return f"Template file '{filename}' not found"
    
    def generate_feedback_prompt(self, em_score: float, chrf_score: float, bert_score: float) -> str:
        feedback_prompt = ""
        
        # EM 피드백
        if em_score < self.em_threshold:
            em_template = self._load_feedback_template("feedback_em.txt")
            em_feedback = em_template.format(score=em_score, threshold=self.em_threshold)
            feedback_prompt += em_feedback + "\n\n"
        
        # chrF 피드백
        if chrf_score < self.chrf_threshold:
            chrf_template = self._load_feedback_template("feedback_chrf.txt")
            chrf_feedback = chrf_template.format(score=chrf_score, threshold=self.chrf_threshold)
            feedback_prompt += chrf_feedback + "\n\n"
        
        # BERTScore 피드백
        if bert_score < self.bert_threshold:
            bert_template = self._load_feedback_template("feedback_bertscore.txt")
            bert_feedback = bert_template.format(score=bert_score, threshold=self.bert_threshold)
            feedback_prompt += bert_feedback + "\n\n"
        
        return feedback_prompt.strip()
    
    def is_passing(self, em_score: float, chrf_score: float, bert_score: float) -> bool:
        return (em_score >= self.em_threshold and 
                chrf_score >= self.chrf_threshold and 
                bert_score >= self.bert_threshold)
    
    def get_threshold_info(self) -> Dict[str, float]:
        return {
            'em_threshold': self.em_threshold,
            'chrf_threshold': self.chrf_threshold,
            'bert_threshold': self.bert_threshold
        }
    
    def update_thresholds(self, em: float = None, chrf: float = None, bert: float = None):
        if em is not None:
            self.em_threshold = em
        if chrf is not None:
            self.chrf_threshold = chrf
        if bert is not None:
            self.bert_threshold = bert