from openai.embeddings_utils import get_embedding
from main import readtestfile

from util import EMBEDDING_MODEL


def EMBEDDING(sentence):
    return get_embedding(sentence, engine=EMBEDDING_MODEL)

if __name__ == "__main__":
    dat = readtestfile("data/English/LREC_expert_label/dev.p")
    dat = dat["tweet"]
    # print(dat)
    # print(type(dat))
    new_dat = dat.apply(EMBEDDING)
    new_dat.to_csv("data/English/LREC_expert_label/dev.csv", index=False)
    # label_embeddings = [get_embedding(label, engine=EMBEDDING_MODEL) for label in labels]