import sys,os,math,random
import torch
import torch.nn as nn

def digital_format_on_list(x, len=6):
    return [round(a, len) for a in x]

def load(file):
    model_dict = torch.load(file)
    #output_file = file + ".param"
    #fo = open(output_file, "r")
    print("model_dict:", model_dict)
    #fo.write("model_dict:\t" + str(model_dict))
    """node_embs = model_dict['node_emb.weight'].cpu().tolist()
    for i, e in enumerate(node_embs):
        print(str(i) + "\t" + " ".join(list(map(str, digital_format_on_list(e, 6)))))
        sys.stdout.flush()
    """

def main():
    file = sys.argv[1]
    load(file)

main()
