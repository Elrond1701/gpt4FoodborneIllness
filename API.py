import json


def getApiKey():
    openai_key_file = 'openaiKey.json'
    with open(openai_key_file, 'r', encoding='utf-8') as f:
        openai_key = json.loads(f.read())
    return openai_key['api']