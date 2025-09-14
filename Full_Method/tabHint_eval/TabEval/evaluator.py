# Table Evaluator
# EM, chrF, BERTScore based table evaluation module

import pandas as pd
import numpy as np
import json
import os
import sys
import re
from pathlib import Path
import tempfile
from typing import Dict, List, Tuple, Optional
import bert_score
from sacrebleu import sentence_chrf
import torch


class TableEvaluator:
    def __init__(self, use_cuda: bool = True, gpu_id: int = None):
        if use_cuda and torch.cuda.is_available():
            if gpu_id is not None:
                self.device = f'cuda:{gpu_id}'
            else:
                gpu_id = self._select_best_gpu()
                self.device = f'cuda:{gpu_id}' if gpu_id is not None else 'cpu'
        else:
            self.device = 'cpu'
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        self.bert_scorer = bert_score.BERTScorer(
            model_type='roberta-large', 
            lang='en', 
            rescale_with_baseline=True, 
            device=self.device,
            batch_size=1
        )
        
        self.metric_cache = {}
    
    def _select_best_gpu(self):
        """select the most free GPU"""
        if not torch.cuda.is_available():
            return None
        
        try:
            import subprocess
            import re

            result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                print("Warning: nvidia-smi failed, using GPU 0")
                return 0
                
            gpu_info = []
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) >= 3:
                    gpu_id = int(parts[0])
                    used_mem = int(parts[1])
                    total_mem = int(parts[2])
                    usage_ratio = used_mem / total_mem
                    gpu_info.append((gpu_id, usage_ratio, used_mem))
            
            if gpu_info:
                best_gpu = min(gpu_info, key=lambda x: x[1])
                print(f"Selected GPU {best_gpu[0]} (usage: {best_gpu[1]:.1%}, used: {best_gpu[2]}MB)")
                return best_gpu[0]
                
        except Exception as e:
            print(f"Warning: GPU selection failed ({e}), using GPU 0")
            
        return 0
    
    def parse_table_from_markdown(self, text: str) -> np.ndarray:
        lines = text.strip().split('\n')
        
        table_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith('|') and line.endswith('|'):
                table_lines.append(line)
        
        if not table_lines:
            return np.array([[]])
        
        content_lines = []
        for line in table_lines:
            if not re.match(r'^\s*\|[\s\-\|:]+\|\s*$', line):
                content_lines.append(line)
        
        if not content_lines:
            return np.array([[]])
        
        data = []
        for line in content_lines:
            cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
            cells = [cell if cell != '' else None for cell in cells]
            data.append(cells)
        
        if data:
            max_cols = max(len(row) for row in data)
            for row in data:
                while len(row) < max_cols:
                    row.append(None)
        
        return np.array(data, dtype=object)
    
    def to_python_str(self, s):
        if isinstance(s, np.ndarray):
            return ' '.join(map(str, s.flatten()))
        elif isinstance(s, (np.str_, np.generic)):
            return s.item()
        elif s is None:
            return ''
        else:
            return str(s)
    
    def calc_similarity(self, tgt, pred, metric):
        if isinstance(tgt, tuple) and isinstance(pred, tuple):
            if len(tgt) != len(pred):
                return 0.0
            sim = 1.0
            for t_elem, p_elem in zip(tgt, pred):
                elem_sim = self.calc_similarity(t_elem, p_elem, metric)
                sim *= elem_sim
            return sim
        else:
            tgt_str = self.to_python_str(tgt)
            pred_str = self.to_python_str(pred)
            
            cache_key = (tgt_str, pred_str, metric)
            if cache_key in self.metric_cache:
                return self.metric_cache[cache_key]
            
            if metric == 'exact_match':
                sim = float(tgt_str.strip() == pred_str.strip())
            elif metric == 'chrf':
                sim = sentence_chrf(pred_str, [tgt_str]).score / 100
            elif metric == 'BERT_score':
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                max_len = 256
                if len(pred_str) > max_len:
                    pred_str = pred_str[:max_len]
                if len(tgt_str) > max_len:
                    tgt_str = tgt_str[:max_len]
                
                P, R, F1 = self.bert_scorer.score([pred_str], [tgt_str])
                sim = F1.item()
                sim = max(0.0, min(sim, 1.0))
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            self.metric_cache[cache_key] = sim
            return sim
    
    def parse_table_to_data(self, table):
        if table.ndim != 2 or table.size == 0:
            return set(), set()
        
        num_rows, num_cols = table.shape
        
        has_index_col = False
        if num_rows > 2 and num_cols > 1:
            try:
                first_col_vals = []
                for i in range(1, min(4, num_rows)):
                    val = self.to_python_str(table[i, 0]).strip()
                    if val.isdigit():
                        first_col_vals.append(int(val))
                    else:
                        break
                
                if len(first_col_vals) >= 2 and first_col_vals[0] == 0 and first_col_vals == list(range(len(first_col_vals))):
                    has_index_col = True
            except:
                pass
        
        start_col = 1 if has_index_col else 0
        
        col_headers = set()
        relations = set()
        
        if num_rows > 0 and num_cols > start_col:
            for j in range(start_col, num_cols):
                col_header = self.to_python_str(table[0, j])
                if col_header:
                    col_headers.add(col_header)
        
        for i in range(1, num_rows):
            for j in range(start_col, num_cols):
                cell_value = table[i, j]
                
                if cell_value in ['', None]:
                    continue
                if isinstance(cell_value, float) and np.isnan(cell_value):
                    continue
                
                cell_str = self.to_python_str(cell_value)
                if cell_str:
                    relation = (i-1, j-start_col, cell_str)
                    relations.add(relation)
        
        return col_headers, relations
    
    def metrics_by_sim(self, tgt_data, pred_data, metric_name):
        if not tgt_data and not pred_data:
            return 1.0, 1.0, 1.0
        if not pred_data:
            return 0.0, 0.0, 0.0
        if not tgt_data:
            return 0.0, 0.0, 0.0
        
        tgt_data_list = list(tgt_data)
        pred_data_list = list(pred_data)
        
        sim_matrix = np.zeros((len(tgt_data_list), len(pred_data_list)), dtype=float)
        for i, tgt_item in enumerate(tgt_data_list):
            for j, pred_item in enumerate(pred_data_list):
                sim = self.calc_similarity(tgt_item, pred_item, metric_name)
                sim_matrix[i, j] = sim
        
        max_sim_pred = np.max(sim_matrix, axis=0)
        precision = np.mean(max_sim_pred)
        
        max_sim_tgt = np.max(sim_matrix, axis=1)
        recall = np.mean(max_sim_tgt)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return precision, recall, f1
    
    def evaluate_table(self, model_output, gold_label):
        gold_col_headers, gold_relations = self.parse_table_to_data(gold_label)
        model_col_headers, model_relations = self.parse_table_to_data(model_output)
        
        metrics = {}
        metric_names = ['exact_match', 'chrf', 'BERT_score']
        
        cells_metrics = {}
        for metric_name in metric_names:
            precision, recall, f1 = self.metrics_by_sim(
                gold_relations, model_relations, metric_name
            )
            cells_metrics[metric_name + '(%)'] = {
                'precision': precision * 100,
                'recall': recall * 100,
                'f1': f1 * 100,
            }
        metrics['cells'] = cells_metrics
        
        col_header_metrics = {}
        for metric_name in metric_names:
            precision, recall, f1 = self.metrics_by_sim(
                gold_col_headers, model_col_headers, metric_name
            )
            col_header_metrics[metric_name + '(%)'] = {
                'precision': precision * 100,
                'recall': recall * 100,
                'f1': f1 * 100,
            }
        metrics['col_header'] = col_header_metrics
        
        return metrics

    def evaluate_table_pair(self, original_table: str, generated_table: str, title: str = "") -> dict:
        try:
            gold_table = self.parse_table_from_markdown(original_table)
            generated_table_array = self.parse_table_from_markdown(generated_table)
            
            metrics = self.evaluate_table(generated_table_array, gold_table)
            
            return self._parse_evaluation_result({'metrics': metrics})
                        
        except Exception as e:
            return {
                'success': False,
                'scores': {},
                'detailed_metrics': {},
                'error': f'Evaluation failed: {str(e)}'
            }
    
    def _parse_evaluation_result(self, result: dict) -> dict:
        try:
            # extract scores
            cells_metrics = result['metrics'].get('cells', {})
            
            em_f1 = cells_metrics.get('exact_match(%)', {}).get('f1', 0.0)
            chrf_f1 = cells_metrics.get('chrf(%)', {}).get('f1', 0.0)
            bert_f1 = cells_metrics.get('BERT_score(%)', {}).get('f1', 0.0)
            
            return {
                'success': True,
                'scores': {
                    'em_f1': em_f1 / 100.0,  # 0-1 range normalization
                    'chrf_f1': chrf_f1 / 100.0,
                    'bert_f1': bert_f1 / 100.0
                },
                'detailed_metrics': result['metrics'],
                'error': ''
            }
        except Exception as e:
            return {
                'success': False,
                'scores': {},
                'detailed_metrics': {},
                'error': f'Failed to parse evaluation result: {str(e)}'
            }
    
    def get_score_breakdown(self, scores: dict) -> dict:
        if not scores:
            return {'analysis': 'No scores available'}
        
        em = scores.get('em_f1', 0)
        chrf = scores.get('chrf_f1', 0)
        bert = scores.get('bert_f1', 0)
        
        analysis = {
            'em_contribution': em * 0.2,
            'chrf_contribution': chrf * 0.4,
            'bert_contribution': bert * 0.4,
            'total_weighted': (em * 0.2) + (chrf * 0.4) + (bert * 0.4),
            'strongest_metric': max([('EM', em), ('chrF', chrf), ('BERT', bert)], key=lambda x: x[1])[0],
            'weakest_metric': min([('EM', em), ('chrF', chrf), ('BERT', bert)], key=lambda x: x[1])[0]
        }
        
        return analysis
