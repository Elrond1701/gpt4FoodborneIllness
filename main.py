import argparse
import json
import os
import pickle
import openai

from API import getApiKey
openai.api_key = getApiKey()

from sklearn.metrics import classification_report, confusion_matrix
from EMD import EMD
from SF import SF

from TRC import TRC, label_embeddings


# os.environ["HTTP_PROXY"] = proxy
# os.environ["HTTPS_PROXY"] = proxy


def oneTask(sentence: str, task: str, examples=False):
    if task == "TRC":
        return TRC(sentence, label_embeddings)
    elif task == "EMD":
        return EMD(sentence, examples)
    elif task == "SF":
        return SF(sentence, examples)
    else:
        raise Exception()
    

def readtestfile(file):
    file = open(file, "rb")
    return pickle.load(file)


def test(args):
    assert args.file is not None
    dat = readtestfile(args.file)
    dat = dat.rename(columns={'tweet': 'sentence'})
    true = []
    pred = []
    i = 0
    for _, data in dat.iterrows():
        res = oneTask(data["sentence"], args.task)
        if args.task == "TRC":
            true.append(data["sentence_class"])
            pred.append(1 if res == "relevant" else 0)
        elif args.task == "EMD":
            true.append(data[""])
            pred.append(res)
        elif args.task == "SF":
            true.append(data[""])
            pred.append(res)
        i += 1
        print("Times: " + str(i) + ", _: " + str(_))
        if i >= 50:
            break
    p = classification_report(y_true=true, y_pred=pred)
    print(p)
    print(true)
    print(pred)
    return true, pred


def show(args):
    while True:
        sentence = input("Please input a sentence: ")
        if sentence == "finish":
            break
        res = oneTask(sentence, args.task)
        print(args.task + ": " + res)


def predict(args):
    assert args.file is not None
    dat = readtestfile(args.file)
    dat = dat.rename(columns={'tweet': 'sentence'})
    pred = []
    for _, data in dat.iterrows():
        res = oneTask(data["sentence"], args.task)
        if args.task == "TRC":
            pred.append({"sentence": data["sentence"], 
                         "pred": 1 if res == "relevant" else 0
                        })
        elif args.task == "EMD":
            pred.append({"sentence": data["sentence"], "pred": res})
        elif args.task == "SF":
            pred.append({"sentence": data["sentence"], 
                         "pred": res
                        })
    print(pred)
    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="show", type=str)
    parser.add_argument("--task", default="TRC", type=str)
    parser.add_argument("--file", default=None, type=str)
    args = parser.parse_args()
    
    if args.type == "show":
        show(args)
    elif args.type == "test":
        test(args)
    elif args.type == "predict":
        predict(args)
    else:
        raise Exception()
