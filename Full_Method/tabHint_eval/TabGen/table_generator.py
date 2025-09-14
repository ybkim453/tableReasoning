import os
import re
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CheckTableGenerator:
    
    def __init__(self, model_name="gpt-4o"):
        self.client = OpenAI()
        self.model_name = model_name
        self.prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
    
    def load_prompt(self, prompt_name):
        """Load prompt from file"""
        prompt_path = os.path.join(self.prompts_dir, f"{prompt_name}.txt")
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"Prompt file not found: {prompt_path}")
            return None
    
    def parse_input_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract components
        idx_match = re.search(r'=================== idx & table_id ===================\n(.*?)\n', content)
        idx_info = idx_match.group(1).strip() if idx_match else ""
        
        title_match = re.search(r'=================== Title ===================\n(.*?)\n', content)
        title = title_match.group(1).strip() if title_match else ""
        
        summary_match = re.search(r'=================== Table Summary ===================\n(.*?)(?=\n===================|\Z)', content, re.DOTALL)
        table_summary = summary_match.group(1).strip() if summary_match else ""
        
        return idx_info, title, table_summary
    
    def step1_extract_schema(self, title, table_summary, log_file_path):
        prompt_template = self.load_prompt("Schema_Extraction")
        if not prompt_template:
            return None
        
        prompt = f"{prompt_template}\n\n***INPUT***:\nTitle: {title}\n\nTable Summary:\n{table_summary}\n\nExtract the schema:"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting table schemas. Always return valid JSON following the exact format specified."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content
            
            # Log the full response
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"STEP 1: SCHEMA EXTRACTION\n")
                f.write(f"{'='*50}\n")
                f.write(f"LLM Response:\n{response_text}\n")
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                schema = json.loads(json_match.group())
                
                # Log schema result
                with open(log_file_path, 'a', encoding='utf-8') as f:
                    f.write(f"\nExtracted Schema:\n{json.dumps(schema, indent=2)}\n")
                
                return schema
            else:
                print("No JSON found in schema response")
                return None
                
        except Exception as e:
            print(f"Error in step 1: {str(e)}")
            return None
    
    def step2_extract_instructions(self, table_summary, log_file_path):
        prompt_template = self.load_prompt("Instruction_Extraction")
        if not prompt_template:
            return None
        
        prompt = f"{prompt_template}\n\n***TABLE SUMMARY***:\n{table_summary}\n\nExtract the instructions:"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at converting table summaries into detailed instructions. Always return valid JSON following the exact format specified."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            response_text = response.choices[0].message.content
            
            # Log the full response
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"STEP 2: INSTRUCTION EXTRACTION\n")
                f.write(f"{'='*50}\n")
                f.write(f"LLM Response:\n{response_text}\n")
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                instructions = json.loads(json_match.group())
                
                # Log instructions result
                with open(log_file_path, 'a', encoding='utf-8') as f:
                    f.write(f"\nExtracted Instructions:\n{json.dumps(instructions, indent=2)}\n")
                
                return instructions
            else:
                print("No JSON found in instructions response")
                return None
                
        except Exception as e:
            print(f"Error in step 2: {str(e)}")
            return None
    
    def step3_generate_table(self, schema, instructions, log_file_path):
        prompt_template = self.load_prompt("Table_Generation")
        if not prompt_template:
            return None
        
        prompt = f"{prompt_template}\n\n***SCHEMA***:\n{json.dumps(schema, indent=2)}\n\n***INSTRUCTIONS***:\n{json.dumps(instructions, indent=2)}\n\nGenerate the table following the step-by-step process:"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert table generator. Generate realistic, coherent tables that exactly match all specifications."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2500
            )
            
            response_text = response.choices[0].message.content
            
            # Log the full response
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"STEP 3: TABLE GENERATION\n")
                f.write(f"{'='*50}\n")
                f.write(f"LLM Response:\n{response_text}\n")
            
            return response_text
            
        except Exception as e:
            print(f"Error in step 3: {str(e)}")
            return None
    
    def process_file(self, input_file_path, output_dir):
        print(f"Processing file with Advanced Information method: {input_file_path}")
        
        # Parse input file
        idx_info, title, table_summary = self.parse_input_file(input_file_path)
        
        print(f"Index Info: {idx_info}")
        print(f"Title: {title}")
        print(f"Table Summary length: {len(table_summary)} characters")
        
        # Create log file
        base_name = os.path.basename(input_file_path).replace('.txt', '')
        log_file_path = os.path.join(output_dir, f"{base_name}_processing_log.txt")
        
        # Initialize log
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.write(f"ADVANCED INFORMATION TABLE GENERATION LOG\n")
            f.write(f"{'='*60}\n")
            f.write(f"Method: Table Generation by Advanced Information - Ours\n")
            f.write(f"File: {input_file_path}\n")
            f.write(f"Title: {title}\n")
            f.write(f"Index Info: {idx_info}\n")
            f.write(f"Processing Time: {os.popen('date').read().strip()}\n")
            f.write(f"\nProcess Flow:\n")
            f.write(f"1. Schema Extraction: Title + Table Summary → Table name + Column headers\n")
            f.write(f"2. Instruction Extraction: Table Summary → Detailed generation instructions\n")
            f.write(f"3. Table Generation: Schema + Instructions → Complete table\n")
        
        # Step 1: Extract Schema
        print("\nStep 1: Extracting table schema...")
        schema = self.step1_extract_schema(title, table_summary, log_file_path)
        if not schema:
            print("Failed at Step 1 - Schema Extraction")
            return None
        
        # Save step 1 output
        step1_file = os.path.join(output_dir, f"{base_name}_step1_schema.json")
        with open(step1_file, 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=2)
        print(f"Step 1 completed, saved to: {step1_file}")
        
        # Step 2: Extract Instructions
        print("\nStep 2: Extracting generation instructions...")
        instructions = self.step2_extract_instructions(table_summary, log_file_path)
        if not instructions:
            print("Failed at Step 2 - Instruction Extraction")
            return None
        
        # Save step 2 output
        step2_file = os.path.join(output_dir, f"{base_name}_step2_instructions.json")
        with open(step2_file, 'w', encoding='utf-8') as f:
            json.dump(instructions, f, indent=2)
        print(f"Step 2 completed, saved to: {step2_file}")
        
        # Step 3: Generate Table
        print("\nStep 3: Generating table")
        generated_table = self.step3_generate_table(schema, instructions, log_file_path)
        if not generated_table:
            print("Failed at Step 3 - Table Generation")
            return None
        
        # Save final output
        step3_file = os.path.join(output_dir, f"{base_name}_step3_generated_table.md")
        with open(step3_file, 'w', encoding='utf-8') as f:
            f.write(f"# Table Generation by Advanced Information - Ours\n\n")
            f.write(f"**Title:** {title}\n")
            f.write(f"**Source:** {idx_info}\n\n")
            f.write("## Process Overview\n\n")
            f.write("1. **Schema Extraction**: Title + Table Summary → Table name + Column headers\n")
            f.write("2. **Instruction Extraction**: Table Summary → Detailed generation instructions\n")
            f.write("3. **Table Generation**: Schema + Instructions → Complete table\n\n")
            f.write("## Generated Table\n\n")
            f.write(generated_table)
            f.write(f"\n\n## Processing Log\n\n")
            f.write(f"See: {log_file_path}")
        print(f"Step 3 completed, saved to: {step3_file}")
        
        return step3_file

