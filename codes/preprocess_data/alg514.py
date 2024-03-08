#http://groups.csail.mit.edu/rbg/code/wordprobs/questions.json
# Nate Kushman, Yoav Artzi, Luke Zettlemoyer, and Regina Barzilay. 2014. Learning to Automatically Solve Algebra Word Problems. In Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 271–281, Baltimore, Maryland. Association for Computational Linguistics.

#仅保留只有一个输出的问题。按照8:1:2的比例随机划分数据集
import json,random

with open("datasets/Alg514/alg514_questions.json") as f:
    data=json.load(f)

new_data=[]
for d in data:
    if len(d["lSolutions"])==1:
        new_data.append(d)

random.seed(202401261353)
random.shuffle(new_data)

train_val_test_list=[8,1,2]
tvt_sum=sum(train_val_test_list)
tvt_ratio_list=[i/tvt_sum for i in train_val_test_list]
train_end_index=int(tvt_ratio_list[0]*len(new_data))
val_end_index=train_end_index+int(tvt_ratio_list[1]*len(new_data))

json.dump(new_data[:train_end_index],open("datasets/Alg514/alg514_questions_1result_train.json","w"),
          ensure_ascii=False,indent=4)
json.dump(new_data[train_end_index:val_end_index],open("datasets/Alg514/alg514_questions_1result_valid.json","w"),
          ensure_ascii=False,indent=4)
json.dump(new_data[val_end_index:],open("datasets/Alg514/alg514_questions_1result_test.json","w"),
          ensure_ascii=False,indent=4)