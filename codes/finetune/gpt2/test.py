# 本代码用于调用本地储存的checkpoint进行测试

# 代码参考https://github.com/openai/grade-school-math/blob/master/grade_school_math/sample.py

from hide_config import gpt_2_english_path, gpt_2_chinese_path

import sys

sys.path.append("codes")


import os, json
from tqdm import tqdm

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from get_data import get_data
from extract_number import extract_last_number


num_threads = "32"
torch.set_num_threads(int(num_threads))
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-dn", "--dataset_name", type=str)
# 目前支持的选项：
# Alg514 GSM8K

parser.add_argument("-ds", "--dataset_path", type=str)
# 储存数据的文件夹

parser.add_argument("--language", type=str, default="en")

parser.add_argument("-cp", "--checkpoint_path", type=str)

parser.add_argument("-rt", "--result_path", type=str)  # 输出测试结果的文本文件

args = parser.parse_args()
arg_dict = args.__dict__


all_data, threshold = get_data(arg_dict["dataset_name"], arg_dict["dataset_path"])

test_examples = all_data["test"]
for example in test_examples:
    example.update(question=example["question"] + "\n")

print(f"{len(test_examples)} test examples")

if arg_dict["language"] == "en":
    gpt2path = gpt_2_english_path
else:
    gpt2path = gpt_2_chinese_path

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

tokenizer = GPT2Tokenizer.from_pretrained(gpt2path)
model = GPT2LMHeadModel.from_pretrained(arg_dict["checkpoint_path"])
model.to(device)
print("Model Loaded")


write_file = open(arg_dict["result_path"], "a")

amount_predict_right = 0
with torch.no_grad():
    for sample in tqdm(test_examples):
        question_prompt = sample["question"] + " The answer is:"
        toks = tokenizer(question_prompt, padding=False, return_tensors="pt").to(device)

        outputs = model.generate(
            **toks,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=model.config.eos_token_id,
        )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_text = output_text[
            output_text.find(question_prompt) + len(question_prompt) :
        ]

        predict_value = extract_last_number(output_text)
        if predict_value is None:
            predict_value = 0

        if abs(predict_value - sample["answer"]) < threshold:
            amount_predict_right += 1

        write_file.write(
            json.dumps(
                {"model_prediction": output_text, "predict_value": predict_value},
                ensure_ascii=False,
            )
            + "\n"
        )
        write_file.flush()

print(amount_predict_right)
print(amount_predict_right / len(test_examples))
