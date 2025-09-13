import openai
import os
import time
import tiktoken
import timeout_decorator
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Union
            
class Model:
    def __init__(self, model_name: str, provider: str = 'openai'):
        self.model_name = model_name
        self.provider = provider  # 'openai' or 'huggingface'
        if provider == 'huggingface':
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        elif provider == 'openai':
            self.tokenizer = tiktoken.encoding_for_model(model_name)
            
            API_KEY = os.getenv("OPENAI_API_KEY", None)
            
            if API_KEY is None:
                raise ValueError("OPENAI_API_KEY not set, please run `export OPENAI_API_KEY=<your key>` to ser it")
            else:
                openai.api_key = API_KEY

    def query(self, messages=None, prompt: str = None, **kwargs):
        if self.provider == 'openai':
            return self.query_openai(messages=messages, prompt=prompt, **kwargs)
        elif self.provider == 'huggingface':
            return self.query_huggingface(prompt, **kwargs)
        else:
            raise ValueError("Unsupported provider")

    @timeout_decorator.timeout(60, timeout_exception=StopIteration)
    def query_with_timeout(self, messages, **kwargs):
        return openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )


    def query_openai(
    self,
    prompt: Optional[str] = None,
    messages: Optional[list] = None,
    system: Optional[str] = None,
    rate_limit_per_minute: Optional[int] = None,
    **kwargs
    ):
        # ✅ 메시지 우선
        if messages is not None:
            msgs = messages
        elif system is None:
            msgs = [{"role": "user", "content": prompt}]
        else:
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

        for i in range(64):
            try:
                response = self.query_with_timeout(msgs, **kwargs)

                if rate_limit_per_minute:
                    time.sleep(60 / rate_limit_per_minute - 0.5)

                if kwargs.get("n", 1) == 1:
                    return response.choices[0].message["content"], response
                else:
                    return [choice.message["content"] for choice in response.choices], response

            except StopIteration:
                print("Query timed out, retrying...")
                continue
            except Exception as e:
                print(e)
                time.sleep(10)

        raise RuntimeError("Failed to query the OpenAI API after 64 retries.")

    def query_huggingface(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        outputs = self.model.generate(inputs, **kwargs)

        # Decode the generated text
        decoded_outputs = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the start of the sequence
        prompt_length = len(self.tokenizer.decode(inputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
        return decoded_outputs[prompt_length:], {"prompt": prompt, "prompt_length": len(inputs[0])}
        