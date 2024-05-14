from hide_config import *
from typing import Callable
import os


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

    # 类OpenAI接口
    elif model_checkpoint_path in [
        "yi-large",
        "yi-large-preview",
        "deepseek-chat",
        "moonshot-v1-8k",
        "GLM-4",
    ]:
        temperature = 0.9

        # 调用openai SDK
        if model_checkpoint_path in [
            "yi-large",
            "yi-large-preview",
            "deepseek-chat",
            "moonshot-v1-8k",
        ]:
            from openai import OpenAI

            if model_checkpoint_path in ["yi-large", "yi-large-preview"]:
                API_BASE = "https://api.lingyiwanwu.com/v1"
                API_KEY = Yi_key
            elif model_checkpoint_path == "deepseek-chat":
                API_BASE = "https://api.deepseek.com"
                API_KEY = DeepSeek_KEY
                temperature = 0
            else:
                API_BASE = "https://api.moonshot.cn/v1"
                API_KEY = Moonshot_KEY
                temperature = 0.3
            client = OpenAI(api_key=API_KEY, base_url=API_BASE)

            local_model_name = model_checkpoint_path

        # 调用原生SDK
        elif model_checkpoint_path == "GLM-4":
            from zhipuai import ZhipuAI

            client = ZhipuAI(api_key=Zhipu_key)

            local_model_name = "glm-4"

        def predict(content):
            response = client.chat.completions.create(
                model=local_model_name,
                messages=[{"role": "user", "content": content}],
                top_p=0.7,
                temperature=temperature,
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
    elif model_name == "ChatGLM3":
        from transformers import AutoModel, AutoTokenizer

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
            response, _ = model.chat(tokenizer, content, history=[], temperature=0.01)
            return response

    elif model_name == "Meta-Llama-3-8B-Instruct":
        # 我只写了LLaMA3原生模型权重的版本
        import sys

        sys.path.append(
            llama_module_path
        )  # https://github.com/meta-llama/llama3 下载到本地的路径。因为要调用这里面的llama模块

        from llama import Dialog, Llama

        from typing import List

        generator = Llama.build(
            ckpt_dir=model_checkpoint_path,
            tokenizer_path=os.path.join(model_checkpoint_path, "tokenizer.model"),
            max_seq_len=8192,
            max_batch_size=6,
        )

        def predict(content, max_length=1024):
            dialogs: List[Dialog] = [[{"role": "user", "content": content}]]
            results = generator.chat_completion(
                dialogs,
                max_gen_len=max_length,  # 其实这地方不该这么写的，写成max_length-calculate_length(input_str)比较合适
                temperature=0.6,
                top_p=0.9,
            )
            for _, result in zip(dialogs, results):
                return result["generation"]["content"]

    return predict
