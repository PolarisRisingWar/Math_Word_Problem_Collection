from hide_config import gpt_2_english_path, gpt_2_chinese_path

import sys

sys.path.append("codes")

import os

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from datasets import Dataset

from get_data import get_data

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

parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("-bs", "--batch_size", type=int, default=16)
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5)
parser.add_argument("-e", "--num_epochs", type=int, default=20)
parser.add_argument("-ws", "--warmup_steps", type=int, default=0)
parser.add_argument("-np", "--num_proc", type=int, default=4)

args = parser.parse_args()
arg_dict = args.__dict__

max_seq_length = arg_dict["max_length"]
num_proc = arg_dict["num_proc"]

###构建分词器
if arg_dict["language"] == "en":
    gpt2path = gpt_2_english_path
else:
    gpt2path = gpt_2_chinese_path

tokenizer = GPT2Tokenizer.from_pretrained(gpt2path)
tokenizer.pad_token = tokenizer.eos_token


###构建数据集
all_data, threshold = get_data(arg_dict["dataset_name"], arg_dict["dataset_path"])


def tokenize_function(examples):
    examples["text"] = [
        line for line in examples["text"] if len(line) > 0 and not line.isspace()
    ]
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_length,
    )


dataset_dict = {}
for dataset_split in ["train", "valid"]:
    if dataset_split in all_data:
        examples = all_data[dataset_split]
        example_dataset = {"text": []}
        for example in examples:
            example_dataset["text"].append(
                example["question"]
                + " The answer is:\n"
                + example["answer_with_reasoning"]
                + "<|endoftext|>"
            )

        dataset = Dataset.from_dict(example_dataset)

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=["text"],
        )

        dataset_dict[dataset_split] = tokenized_dataset

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


###加载模型
model = GPT2LMHeadModel.from_pretrained(gpt2path)

training_args = TrainingArguments(
    output_dir=arg_dict["checkpoint_path"],
    overwrite_output_dir=True,
    num_train_epochs=arg_dict["num_epochs"],
    per_device_train_batch_size=arg_dict["batch_size"],
    per_device_eval_batch_size=arg_dict["batch_size"],
    eval_steps=400,
    save_steps=800,
    warmup_steps=arg_dict["warmup_steps"],
    prediction_loss_only=True,
    report_to="none",
)

trainer_args = {
    "model": model,
    "args": training_args,
    "data_collator": data_collator,
    "train_dataset": dataset_dict["train"],
}
if "valid" in dataset_dict:
    trainer_args["eval_dataset"] = dataset_dict["valid"]

trainer = Trainer(**trainer_args)
trainer.train()

trainer.save_model()
