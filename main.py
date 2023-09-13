import argparse
import openai

from API import getApiKey
from model import SVM, DecisionTree, GradientBoosting, LogisticRegression, RandomForest
from util import readfile
openai.api_key = getApiKey()

from sklearn.metrics import classification_report
from EMD import EMD
from SF import SF

from TRC import TRC, label_embeddings


# os.environ["HTTP_PROXY"] = proxy
# os.environ["HTTPS_PROXY"] = proxy


def inference(sentence: str, task: str, examples=False, model=None):
    if task == "TRC":
        return TRC(sentence, model=model)
    elif task == "EMD":
        return EMD(sentence, examples)
    elif task == "SF":
        raise NotImplementedError
        return SF(sentence, examples)
    else:
        raise Exception()


def train(args):
    if args.train_file is None:
        return None
    
    dat = readfile(args.train_file)
    if args.train_model == "RF":
        model = RandomForest
    elif args.train_model == "GB":
        model = GradientBoosting()
    elif args.train_model == "LR":
        model = LogisticRegression()
    elif args.train_model == "SVM":
        model = SVM()
    elif args.train_model == "DT":
        model = DecisionTree()
    else:
        raise Exception()
    model.train(dat)
    return model


def test(args):
    assert args.test_file is not None
    dat = readfile(args.test_file)
    dat = dat.rename(columns={'tweet': 'sentence'})
    true = []
    pred = []
    for _, data in dat.iterrows():
        res = inference(data["sentence"], args.task, model=model)
        if args.task == "TRC":
            true.append(data["sentence_class"])
            pred.append(1 if res == "relevant" else 0)
        elif args.task == "EMD":
            true.append(data[""])
            pred.append(res)
        elif args.task == "SF":
            true.append(data[""])
            pred.append(res)
    p = classification_report(y_true=true, y_pred=pred)
    print(p)
    print(true)
    print(pred)
    return true, pred


def show(args, model):
    while True:
        sentence = input("Please input a sentence: ")
        if sentence == "finish":
            break
        res = inference(sentence, args.task, model=model)
        print(args.task + ": " + res)


def predict(args, model):
    assert args.file is not None
    dat = readfile(args.file)
    dat = dat.rename(columns={'tweet': 'sentence'})
    pred = []
    for _, data in dat.iterrows():
        res = inference(data["sentence"], args.task, model=model)
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
    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--train_model", default="RF", type=str)
    parser.add_argument("--test_file", default=None, type=str)
    args = parser.parse_args()
    
    model = train(args)
    if args.type == "show":
        show(args, model)
    elif args.type == "test":
        test(args, model)
    elif args.type == "predict":
        predict(args, model)
    else:
        raise Exception()
