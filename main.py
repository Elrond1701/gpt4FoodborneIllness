import argparse
import json
import os
import pickle
import openai
from sklearn.metrics import confusion_matrix
from EMD import EMD
from SF import SF

from TRC import TRC


# os.environ["HTTP_PROXY"] = proxy
# os.environ["HTTPS_PROXY"] = proxy


def getApiKey():
    openai_key_file = 'openaiKey.json'
    with open(openai_key_file, 'r', encoding='utf-8') as f:
        openai_key = json.loads(f.read())
    return openai_key['api']

def oneTask(sentence: str, task: str, examples=False):
    if task == "TRC":
        answer = TRC(sentence)
        return 1 if answer == "relevant" else 0
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
    confusion_matrix(y_true=true, y_pred=pred)
    return true, pred


def show(args):
    while True:
        sentence = input("Please input a sentence: ")
        if sentence == "finish":
            break
        res = oneTask(sentence, args.task)
        print(args.task + ": " + str(res))


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
    openai.api_key = getApiKey()
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
