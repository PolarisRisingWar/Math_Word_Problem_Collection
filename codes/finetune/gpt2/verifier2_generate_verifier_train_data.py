# 本代码用于通过generator生成verifier的训练数据

from hide_config import gpt_2_english_path, gpt_2_chinese_path

import sys

sys.path.append("codes")


import os, json, signal
from tqdm import tqdm
from contextlib import contextmanager

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

test_examples = all_data["train"]
for example in test_examples:
    example.update(question=example["question"] + "\n")

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

EQUALS_TOKENS = set([28, 796, 47505])


# 来源的来源：https://stackoverflow.com/questions/492519/timeout-on-a-function-call
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)


def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            return eval(formula)
    except Exception as e:
        signal.alarm(0)
        print(f"Warning: Failed to eval {formula}, exception: {e}")
        return None


def use_calculator(sample):
    if "<<" not in sample:
        return None

    parts = sample.split("<<")
    remaining = parts[-1]
    if ">>" in remaining:
        return None
    if "=" not in remaining:
        return None
    lhs = remaining.split("=")[0]
    lhs = lhs.replace(",", "")
    if any([x not in "0123456789*+-/.()" for x in lhs]):
        return None
    return eval_with_timeout(lhs)


correct_count = 0
with torch.no_grad():
    for sample in tqdm(test_examples):
        question_prompt = sample["question"] + " The answer is:"
        qn = question_prompt

        sample_len = 512
        for _ in range(100):
            qn = question_prompt
            for _ in range(sample_len):
                toks = tokenizer([qn], padding=False, return_tensors="pt").to(device)
                orig_len = toks["input_ids"].shape[1]

                out = model.generate(
                    **toks,
                    max_length=orig_len + 1,
                    pad_token_id=model.config.eos_token_id,
                    do_sample=True,
                    temperature=0.7,
                )
                text = tokenizer.batch_decode(out)[0]

                if out[0, -1].item() in EQUALS_TOKENS:
                    answer = use_calculator(text)
                    if answer is not None:
                        # print("Triggered calculator, answer", answer)
                        text = text + str(answer) + ">>"

                qn = text
                if "<|endoftext|>" in qn:
                    break

            qn = qn[qn.find(question_prompt) + len(question_prompt) :]

            predict_value = extract_last_number(qn)
            if predict_value is None:
                predict_value = 0

            if abs(predict_value - sample["answer"]) < threshold:
                label = 1
                correct_count += 1
            else:
                label = 0

            write_file.write(
                json.dumps(
                    {
                        "question": sample["question"],
                        "model_prediction": qn,
                        "label": label,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            write_file.flush()

write_file.close()
print(correct_count)
print(correct_count / (len(test_examples) * 100))
