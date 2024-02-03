from hide_config import CHATGPT3_5_API_KEY,CHATGPT3_5_base_url

def get_infer(model_name:str,model_checkpoint_path:str):
    if model_checkpoint_path=="GPT-3.5":
        import tiktoken
        from tenacity import (
            retry,
            stop_after_attempt,
            wait_random_exponential,
        )
        from openai import OpenAI

        client = OpenAI(api_key=CHATGPT3_5_API_KEY,base_url=CHATGPT3_5_base_url)

        def num_tokens_from_messages(messages,model):
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
        
        @retry(wait=wait_random_exponential(min=20, max=600),stop=stop_after_attempt(6))
        def predict(content):
            return wrap4predict(content)
        
        def wrap4predict(content):
            messages=[
                    {"role": "user", "content":content}
                ]
            
            if num_tokens_from_messages(messages,"gpt-3.5-turbo")<3096:  #留1000个token给输出
                model_name="gpt-3.5-turbo"
            else:
                model_name="gpt-3.5-turbo-16k"

            completion = client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            return completion.choices[0].message.content
    else:
        from transformers import AutoModel,AutoTokenizer
        if model_name=="ChatGLM3":
            model=AutoModel.from_pretrained(model_checkpoint_path,load_in_8bit=False,trust_remote_code=True,
                                            device_map='auto')
            model.eval()
            tokenizer=AutoTokenizer.from_pretrained(model_checkpoint_path,trust_remote_code=True)
            def predict(content):
                response,_=model.chat(tokenizer,content,history=[],temperature=0.01)
                return response
    
    return predict

