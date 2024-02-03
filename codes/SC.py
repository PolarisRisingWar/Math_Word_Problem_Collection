import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-t","--temperature",type=float,default=0.7)
parser.add_argument("--M",type=int,default=10)

args = parser.parse_args()
arg_dict=args.__dict__