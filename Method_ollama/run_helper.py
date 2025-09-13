from agent.model import Model

def query(model, long_model, messages=None, prompt=None, temperature=0, self_consistency=1):
    """
    Execute a query on the model and handle prompt length for choosing the appropriate model.

    Args:
    - model: The primary model for querying.
    - long_model: The long version of the model for longer prompts.
    - prompt (str): The prompt to query.
    - temperature (float): The temperature setting for the query.
    - self_consistency (int): The number of outputs to generate.

    Returns:
    - Tuple: (text, response)
    """
    
    if messages is not None:
            prompt_str = " ".join([m["content"] for m in messages])
    else:
        prompt_str = prompt or ""
    prompt_length = len(long_model.tokenizer.encode(prompt_str))

    if isinstance(model, Model):
        if prompt_length <= 3328:
            return model.query(messages=messages, temperature=temperature, max_tokens=4000 - prompt_length, n=self_consistency)
        elif prompt_length <= 14592:
            print(f"Prompt length -- {prompt_length} is too long, we use the 16k version.")
            safe_max_tokens = min(4096, 15360 - prompt_length)  # ✅ completion 상한 고려
            return long_model.query(
                messages=messages,
                temperature=temperature,
                max_tokens=safe_max_tokens,
                n=self_consistency
            )
        else:
            if self_consistency == 1:
                return f"Prompt length -- {prompt_length} is too long", {prompt_length: prompt_length}
            else:
                return ["Prompt length -- {prompt_length} is too long"] * self_consistency, {prompt_length: prompt_length}
    else:
        # no short version of the model provided, which means we use the long version for all prompts
        if prompt_length <= 14592:
            return long_model.query(messages=messages, temperature=temperature, max_tokens=15360 - prompt_length, n=self_consistency)
        else:
            if self_consistency == 1:
                return f"Prompt length -- {prompt_length} is too long", {prompt_length: prompt_length}
            else:
                return ["Prompt length -- {prompt_length} is too long"] * self_consistency, {prompt_length: prompt_length}