import random
import re

import pandas
from openai.embeddings_utils import cosine_similarity, get_embedding
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
        embeddings.append(get_embedding(dat.loc[i]["text"], EMBEDDING_MODEL))
    knn = NearestNeighbors(n_neighbors=size)
    knn.fit(embeddings)
    _, row = knn.kneighbors([get_embedding(sentence, EMBEDDING_MODEL)])
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
