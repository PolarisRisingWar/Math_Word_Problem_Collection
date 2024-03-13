# 本代码用于微调GPT-2模型，并将模型储存到本地

# 代码参考https://github.com/openai/grade-school-math/blob/master/grade_school_math/train.py
# https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py

from hide_config import gpt_2_english_path, gpt_2_chinese_path

import sys

sys.path.append("codes")


import torch, os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPT2Config, AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from get_data import get_data
from GPT2DataSet import GPT2DefinedDataset

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

parser.add_argument("--language", type=str, default="en")  # 使用的模型的语言，可选en/zh

parser.add_argument("-cp", "--checkpoint_path", type=str)  # 模型存储路径

parser.add_argument("-bs", "--batch_size", type=int, default=16)
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5)
parser.add_argument("-e", "--num_epochs", type=int, default=20)
parser.add_argument("-ws", "--warmup_steps", type=int, default=0)

args = parser.parse_args()
arg_dict = args.__dict__


all_data, threshold = get_data(arg_dict["dataset_name"], arg_dict["dataset_path"])

train_examples = all_data["train"]
for example in train_examples:
    example.update(question=example["question"] + "\n")
    example.update(answer=example["answer_with_reasoning"] + "<|endoftext|>")

print(f"{len(train_examples)} train examples")

if arg_dict["language"] == "en":
    gpt2path = gpt_2_english_path
else:
    gpt2path = gpt_2_chinese_path

tokenizer = GPT2Tokenizer.from_pretrained(gpt2path)
train_dset = GPT2DefinedDataset(tokenizer, train_examples)

config = GPT2Config.from_pretrained(gpt2path)
model = GPT2LMHeadModel.from_pretrained(gpt2path, config=config)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
model.to(device)

model.train()

train_loader = DataLoader(train_dset, batch_size=arg_dict["batch_size"], shuffle=True)
optim = AdamW(model.parameters(), lr=arg_dict["learning_rate"])

num_epochs = arg_dict["num_epochs"]
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optim,
    num_warmup_steps=arg_dict["warmup_steps"],
    num_training_steps=num_training_steps,
)

pbar = tqdm(range(num_training_steps))
for epoch in range(num_epochs):
    for batch in train_loader:
        optim.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs[0]
        loss.backward()
        optim.step()
        lr_scheduler.step()
        pbar.update(1)
        pbar.set_description(f"train_loss: {loss.item():.5f}")

model.save_pretrained(arg_dict["checkpoint_path"])
