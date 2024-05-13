import argparse, json

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--prediction_path", type=str)
parser.add_argument("-a", "--answer_path", type=str)
parser.add_argument("--threshold", type=float,default=0.001)

args = parser.parse_args()
arg_dict = args.__dict__

predictions = [json.loads(x)["predict_value"] for x in open(arg_dict["prediction_path"]).readlines()]
answers = [x["lSolutions"][0] for x in json.load(open(arg_dict["answer_path"]))]

c = 0
for i in range(len(predictions)):
    if abs(predictions[i] - answers[i]) < arg_dict["threshold"]:
        c += 1
print(c / len(answers))
