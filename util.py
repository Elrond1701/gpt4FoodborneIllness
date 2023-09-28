import json
import pickle
import random
import re
import numpy as np

import pandas
from sklearn.neighbors import NearestNeighbors


DEPT = "department"
DEPT_CHINESE = "部门"
FOOD = "food-related product"
FOOD_CHINESE = "产品"
POLL = "pollutants"
POLL_CHINESE = "污染物"
LAW = "standards and regulations"
LAW_CHINESE = "标准法规"

MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
USER = "user"
ASST = "assistant"

proxy = ""

SEED = 42


def findEntities(sentence):
    pattern = r'\@\@(.*?)\#\#'
    matches = re.findall(pattern, sentence)
    return matches


def randomRetrieval(size=5):
    dat = pandas.read_json("./GPT/data/admin.jsonl", lines=True)
    row = []
    for _ in range(size):
        num = random.randint(0, len(dat) - 1)
        while num in row:
            num = random.randint(0, len(dat) - 1)
        row.append(num)
    return dat.loc[row].reset_index(drop=True)


def kNNRetrieval(sentence, size=5):
    dat = pandas.read_json("./GPT/data/admin.jsonl", lines=True)
    embeddings = []
    for i in range(len(dat) - 1):
        embeddings.append(embedding(dat.iloc[i]['text']))
    knn = NearestNeighbors(n_neighbors=size)
    knn.fit(embeddings)
    _, row = knn.kneighbors([embedding(sentence)])
    return dat.loc[row].reset_index(drop=True)


def textGeneration(dat, entity):
    text = ""
    for _, row in dat.iterrows():
        input = row["text"]
        labels = row["label"]
        output = labelInput(input, labels, entity)
        text += "Input: " + input + "\r\nOutput: " + output + "\r\n"
    return text


def labelInput(input, labels, entity):
    pos = []
    for label in labels:
        if entity == DEPT and label[2] == DEPT_CHINESE:
            pos.append(label[0:2])
        elif entity == FOOD and label[2] == FOOD_CHINESE:
            pos.append(label[0:2])
        elif entity == POLL and label[2] == POLL_CHINESE:
            pos.append(label[0:2])
        elif entity == LAW and label[2] == LAW_CHINESE:
            pos.append(label[0:2])
        else:
            raise Exception()
    output = input
    for i in range(len(pos)):
        output = output[:pos[i][0]] + "@@" + output[pos[i][0]:]
        output = output[:(pos[i][1] + 2)] + "##" + output[(pos[i][1] + 2):]

        for j in range(i, len(pos)):
            if i == j:
                continue
            pos[j][0] += 4
            pos[j][1] += 4
    return output


def readfile(file):
    file = open(file, "rb")
    return pickle.load(file)


def string_to_numpy_array(s):
    return np.array(s.split(), dtype=np.float64)


def load_dat(english=True):
    if english is True:
        with open("./data/English/base.json", "r") as file:
            dat = json.load(file)
    else:
        with open("./data/Chinese/base.json", "r") as file:
            dat = json.load(file)
    return dat