from hide_config import *
from typing import Callable


def get_infer(model_name: str, model_checkpoint_path: str) -> Callable[[str], str]:
    # 调用API
    if model_checkpoint_path == "GPT-3.5":
        import tiktoken
        from tenacity import (
            retry,
            stop_after_attempt,
            wait_random_exponential,
        )
        from openai import OpenAI

        client = OpenAI(api_key=CHATGPT3_5_API_KEY, base_url=CHATGPT3_5_base_url)

        def num_tokens_from_messages(messages, model):
            """Returns the number of tokens used by a list of messages."""
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            num_tokens = 0
            for message in messages:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for _, value in message.items():
                    num_tokens += len(encoding.encode(value))
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens

        @retry(
            wait=wait_random_exponential(min=20, max=600), stop=stop_after_attempt(6)
        )
        def predict(content, max_length=None):
            return wrap4predict(content, max_length)

        def wrap4predict(content, max_length=None):
            messages = [{"role": "user", "content": content}]

            if (
                num_tokens_from_messages(messages, "gpt-3.5-turbo") < 3096
            ):  # 留1000个token给输出
                local_model_name = "gpt-3.5-turbo"
            else:
                local_model_name = "gpt-3.5-turbo-16k"

            completion_params = {"model": local_model_name, "messages": messages}

            if max_length is not None:
                completion_params["max_tokens"] = max_length

            completion = client.chat.completions.create(**completion_params)
            return completion.choices[0].message.content
    
    elif model_checkpoint_path in ["yi-large","GLM-4"]:
        if model_checkpoint_path=="yi-large":
            from openai import OpenAI

            API_BASE = "https://api.lingyiwanwu.com/v1"
            API_KEY = Yi_key
            client = OpenAI(
                api_key=API_KEY,
                base_url=API_BASE
            )

            local_model_name="yi-large"

        elif model_checkpoint_path == "GLM-4":
            from zhipuai import ZhipuAI

            client = ZhipuAI(api_key=Zhipu_key)

            local_model_name="glm-4"

        def predict(content):
            response = client.chat.completions.create(
                model=local_model_name,
                messages=[{"role": "user", "content": content}],
                top_p=0.7,
                temperature=0.9,
                stream=False,
                max_tokens=2000,
            )
            return response.choices[0].message.content

    elif model_checkpoint_path in [
        "llama2-70b-4096",
        "mixtral-8x7b-32768",
        "Gemma-7b-it",
    ]:
        from groq import Groq

        client = Groq(
            api_key=groq_key,
        )

        def predict(content):
            completion = client.chat.completions.create(
                model=model_checkpoint_path,
                messages=[{"role": "user", "content": content}],
                temperature=0.5,
                max_tokens=1024,
                top_p=1,
                stream=True,
                stop=None,
            )

            result_str = ""
            for chunk in completion:
                result_str += chunk.choices[0].delta.content or ""

    # 本地服务
    else:
        from transformers import AutoModel, AutoTokenizer

        if model_name == "ChatGLM3":
            model = AutoModel.from_pretrained(
                model_checkpoint_path,
                load_in_8bit=False,
                trust_remote_code=True,
                device_map="auto",
            )
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(
                model_checkpoint_path, trust_remote_code=True
            )

            def predict(content):
                response, _ = model.chat(
                    tokenizer, content, history=[], temperature=0.01
                )
                return response

    return predict
