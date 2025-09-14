# Subtable Generation module
# Subtable extraction using ColumnSimilarityAnalyzer

import os
import csv
import pandas as pd
from io import StringIO
from subtable_extractor_euclid import ColumnSimilarityAnalyzer


class SubtableGenerator:
    def __init__(self):
        self.analyzer = ColumnSimilarityAnalyzer()
    
    def generate_subtable(self, table_file_path: str, log_dir: str, global_idx: int) -> dict:
        try:
            # Subtable log directory creation
            subtable_dir = os.path.join(log_dir, "subtable_logs")
            os.makedirs(subtable_dir, exist_ok=True)
            
            # ColumnSimilarity based preview generation
            subtable_path = os.path.join(subtable_dir, f"{global_idx}_log.txt")
            result_lines, selected_rows = self.analyzer.process_single_table(table_file_path, subtable_path)
            
            selected_rows = [r + 1 for r in selected_rows]
            
            data_lines = []
            for line in result_lines[2:]:
                fields = line.split('\t')
                is_separator = all(field.strip() and all(c in ':-' for c in field.strip()) for field in fields if field.strip())
                if not is_separator:
                    data_lines.append(line)
            
            csv_content = "\n".join([result_lines[0]] + data_lines)
            
            preview_df = pd.read_csv(
                StringIO(csv_content),
                sep="\t",
                quoting=csv.QUOTE_NONE,
                on_bad_lines="skip"
            )
            
            if "Unnamed: 0" in preview_df.columns:
                preview_df = preview_df.drop(columns=["Unnamed: 0"])
            
            preview_df.columns = [col.replace('\\n', ' ').replace('\n', ' ') for col in preview_df.columns]

            preview_table = preview_df.to_markdown(index=False)
            
            # load full table
            full_df = pd.read_csv(
                table_file_path,
                sep="\t",
                quoting=csv.QUOTE_NONE,
                on_bad_lines="skip"
            )

            if "Unnamed: 0" in full_df.columns:
                full_df = full_df.drop(columns=["Unnamed: 0"])

            full_df.columns = [col.replace('\\n', ' ').replace('\n', ' ') for col in full_df.columns]
            
            full_table = full_df.to_markdown()
            
            with open(subtable_path, "a", encoding="utf-8") as f:
                f.write("\n=================== Generated Subtable ===================\n")
                f.write(preview_table)
                f.write("\n\n=================== Full Table ===================\n")
                f.write(full_table)
                f.write("\n")
            
            return {
                'success': True,
                'preview_table': preview_table,
                'full_table': full_table,
                'selected_rows': selected_rows,
                'subtable_path': subtable_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'preview_table': '',
                'full_table': '',
                'selected_rows': [],
                'subtable_path': ''
            }

