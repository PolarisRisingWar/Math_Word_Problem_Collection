# 本代码用于训练verifier

from hide_config import gpt_2_english_path, gpt_2_chinese_path

import sys, json

sys.path.append("codes")


import torch, os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

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

parser.add_argument("-dp", "--dataset_path", type=str)  # 储存数据的文本文件

parser.add_argument("--language", type=str, default="en")  # 使用的模型的语言，可选en/zh

parser.add_argument("-ip", "--initial_path", type=str)  # generator路径
parser.add_argument("-lsp", "--lm_save_path", type=str)  # verifier LM存储路径
parser.add_argument("-sp", "--save_path", type=str)  # verifier全模型权重存储路径

parser.add_argument("-bs", "--batch_size", type=int, default=8)
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5)
parser.add_argument("-e", "--num_epochs", type=int, default=2)  # 论文值
parser.add_argument("-ws", "--warmup_steps", type=int, default=0)

args = parser.parse_args()
arg_dict = args.__dict__


train_examples = [json.loads(x) for x in open(arg_dict["dataset_path"]).readlines()]
for example in train_examples:
    example.update(question=example["question"] + " The answer is:" + "\n")
    example.update(answer=example["model_prediction"])

print(f"{len(train_examples)} train examples")

if arg_dict["language"] == "en":
    gpt2path = gpt_2_english_path
else:
    gpt2path = gpt_2_chinese_path

tokenizer = GPT2Tokenizer.from_pretrained(gpt2path)
train_dset = GPT2DefinedDataset(tokenizer, train_examples, loss_on_prefix=False)


loss_fn = nn.BCEWithLogitsLoss()


class Verifier(nn.Module):
    def __init__(self):
        super(Verifier, self).__init__()
        self.lm = GPT2LMHeadModel.from_pretrained(
            arg_dict["initial_path"]
        )  # 用generator初始化GPT-2模型

        self.linear = nn.Linear(50257, 1)

    def forward(self, input_ids, attention_mask, labels, two_type_label):
        lm_output = self.lm(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        lm_loss = lm_output[0]
        classifier_output = self.linear(lm_output["logits"]).squeeze(-1)
        two_type_label = (
            two_type_label.unsqueeze(1).expand(-1, classifier_output.size()[1]).float()
        )
        classifier_loss = loss_fn(classifier_output, two_type_label)
        return lm_loss + classifier_loss


model = Verifier()

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
        batch_cuda = {k: v.to(device) for k, v in batch[0].items()}
        outputs = model(
            **batch_cuda,
            labels=batch_cuda["input_ids"],
            two_type_label=batch[1].to(device),
        )
        outputs.backward()
        optim.step()
        lr_scheduler.step()
        pbar.update(1)
        pbar.set_description(f"train_loss: {outputs.item():.5f}")

model.lm.save_pretrained(arg_dict["lm_save_path"])
torch.save(model.state_dict(),arg_dict["save_path"])
