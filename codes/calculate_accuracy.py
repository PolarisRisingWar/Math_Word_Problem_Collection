import argparse, json

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--prediction_path", type=str)
parser.add_argument("-a", "--answer_path", type=str)

args = parser.parse_args()
arg_dict = args.__dict__

predictions = [
    json.loads(x)["predict_value"]
    for x in open(arg_dict["prediction_path"]).readlines()
]
answers = [
    float(x["answer"][x["answer"].find("#### ") + 5 :].replace(",", ""))
    for x in [json.loads(i) for i in open(arg_dict["answer_path"]).readlines()]
]

c = 0
for i in range(len(predictions)):
    if abs(predictions[i] - answers[i]) < 1e-5:
        c += 1
print(c / len(answers))
