# This script runs pyagent on all wtq datasets using gpt-3.5
# - tables are not perturbed
# - resorting stage in NORM is disabled
# - each query will be executed 5 times to do self-consistency

CUDA_VISIBLE_DEVICES=0 python run_agent_ollama.py \
    --model llama3.3:70b --long_model llama3.3:70b \
    --provider openai_ollama --dataset wtq --sub_sample False \
    --perturbation none --use_full_table True --norm True --disable_resort True --norm_cache True \
    --resume 0 --stop_at 1e6 --self_consistency 1 --temperature 0.8 \
    --log_dir output/wtq_agent --cache_dir cache/llama3.3
