from mytools.filetools import fileTool

tool = fileTool()
tool.del_file("./runs/alexnet")
tool.del_file("./runs/vgg")
tool.del_file("./runs/resnet50")
tool.del_file("./runs/resnet152")
tool.del_file("./runs/densenet")