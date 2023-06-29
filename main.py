import json
import os
import openai
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
