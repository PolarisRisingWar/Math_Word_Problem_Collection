import os,json,re
from alive_progress import alive_bar

import torch

from hide_config import *

num_threads = '32'
torch.set_num_threads(int(num_threads))
os.environ['OMP_NUM_THREADS'] = num_threads
os.environ['OPENBLAS_NUM_THREADS'] = num_threads
os.environ['MKL_NUM_THREADS'] = num_threads
os.environ['VECLIB_MAXIMUM_THREADS'] = num_threads
os.environ['NUMEXPR_NUM_THREADS'] = num_threads

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-mc","--model_checkpoint_path",type=str)  #本地：预训练模型checkpoint目录；API：模型名称
#目前支持的选项：
#GPT-3.5 ("gpt-3.5-turbo"和"gpt-3.5-turbo-16k"，根据输入文本长度自动判断)

parser.add_argument("-dn","--dataset_name",type=str)
#目前支持的选项：
#Alg514

parser.add_argument("-ds","--dataset_path",type=str)
#Alg514数据：JSON文件对应的位置

parser.add_argument("-pt","--prompt_template",type=str,default="CoT")
#目前支持的选项：
#CoT: 在问题后面加“Let's think step by step.”
#pure：不加任何东西

parser.add_argument("-rt","--result_txt_path",type=str)  #输出结果储存文件（将直接追加写入）

args = parser.parse_args()
arg_dict=args.__dict__

#数据预处理
#answer是数值类对象
if arg_dict["dataset_name"]=="Alg514":
    test_data=[{"question":x["sQuestion"],"answer":x["lSolutions"][0]} for x in \
               json.load(open(arg_dict["dataset_path"]))]


#构建模型

def extract_result_prompt(question,nl_result):
    """从自然语言格式的结果中抽取出结果
    参考Get an A in Math: Progressive Rectification Prompting"""
    return f"""
        Q: {question} 
        A: {nl_result} 
        Therefore, X (expressed in Arabic numerals and without units) is:
        """

def extract_last_number(text):
    """从自然语言格式的结果中抽取出最后一个数值"""
    # 使用正则表达式找到所有数字（整数或浮点数）
    numbers = re.findall(r'-?\d+\.\d+|-?\d+', text)
    
    # 如果没有找到数字，返回None
    if not numbers:
        return None
    
    # 返回最后一个数字，转换为相应的整数或浮点数
    last_number = numbers[-1]
    if '.' in last_number:
        return float(last_number)
    else:
        return int(last_number)

def question2prompt(question):
    if arg_dict["prompt_template"]=="CoT":
        return question+" Let's think step by step."
    elif arg_dict["prompt_template"]=="pure":
        return question
    


if arg_dict["model_checkpoint_path"]=="GPT-3.5":
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
        
#运行模型
result_file=open(arg_dict["result_txt_path"],"a")

amount_predict_right=0
with alive_bar(len(test_data)) as bar:
    for i in range(len(test_data)):
        model_prediction=predict(question2prompt(test_data[i]["question"]))

        predict_result=predict(extract_result_prompt(test_data[i]["question"],model_prediction))
        try:
            predict_result=float(predict_result)
        except:
            predict_result=extract_last_number(predict_result)
            if predict_result is None:
                predict_result=0

        if abs(predict_result-test_data[i]["answer"])<0.00001:
            amount_predict_right+=1
        
        result_file.write(json.dumps({"model_prediction":model_prediction,"predict_result":predict_result},
                                     ensure_ascii=False)+"\n")
        result_file.flush()

        bar()
        #break

result_file.close()

print(amount_predict_right)
print(amount_predict_right/len(test_data))