# 官方数据集下载地址挂了，但是可以在但是可以在代码文件（https://www.dropbox.com/s/1slbq2fi77fq7wx/Java%20code_mathproblems.zip?dl=1）里面找到Math_Word_DS2.zip文件
# 以下直接使用解压文件中ALL文件夹 - ALL.json文件

#Mohammad Javad Hosseini, Hannaneh Hajishirzi, Oren Etzioni, and Nate Kushman. 2014. Learning to Solve Arithmetic Word Problems with Verb Categorization. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 523–533, Doha, Qatar. Association for Computational Linguistics.

# 按照8:1:2的比例随机划分数据集
import json, random

with open("datasets/AI2/ALL.json") as f:
    data = json.load(f)

random.seed(20240308)
random.shuffle(data)

train_val_test_list = [8, 1, 2]
tvt_sum = sum(train_val_test_list)
tvt_ratio_list = [i / tvt_sum for i in train_val_test_list]
train_end_index = int(tvt_ratio_list[0] * len(data))
val_end_index = train_end_index + int(tvt_ratio_list[1] * len(data))

json.dump(
    data[:train_end_index],
    open("datasets/AI2/train.json", "w"),
    ensure_ascii=False,
    indent=4,
)
json.dump(
    data[train_end_index:val_end_index],
    open("datasets/AI2/valid.json", "w"),
    ensure_ascii=False,
    indent=4,
)
json.dump(
    data[val_end_index:],
    open("datasets/AI2/test.json", "w"),
    ensure_ascii=False,
    indent=4,
)
