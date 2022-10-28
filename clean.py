from mytools.filetools import fileTool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="all")


args = parser.parse_args()
model = args.model

tool = fileTool()

if model=="all":
    tool.del_file("./runs")

else:
    tool.del_file("./runs/{}".format(model))

