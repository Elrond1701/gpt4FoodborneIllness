import time
from openai.embeddings_utils import cosine_similarity, get_embedding

from util import EMBEDDING_MODEL


labels = ["Relevent with foodborne illness.", 
          "Not Relevent with foodborne illness."]
label_embeddings = [get_embedding(label, engine=EMBEDDING_MODEL) for label in labels]
time.sleep(60)


def TRC(sentence: str, label_embeddings):
    def label_score(review_embedding, label_embeddings):
        return cosine_similarity(review_embedding, label_embeddings[1]) - cosine_similarity(review_embedding, label_embeddings[0])
    

    probas = label_score(get_embedding(sentence, engine=EMBEDDING_MODEL), label_embeddings)
    preds = "irrelevant" if probas > 0 else "relevant"

    time.sleep(20)

    return preds
