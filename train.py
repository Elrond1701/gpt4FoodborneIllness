import openai
import pandas
from API import getApiKey
from util import readfile

openai.api_key = getApiKey()

from TRC import embedding


if __name__ == "__main__":
    openai.api_key = getApiKey()
    dat = readfile("data/English/LREC_BSC/train.p")
    dat = pandas.concat([dat["id"], dat["tweet"].apply(embedding), dat["sentence_class"]], axis=1)
    dat.to_csv("data/English/LREC_BSC/train.csv", index=True)
    # new_dat.to_csv("data/English/LREC_expert_label/dev.csv", index=False)