from openai.embeddings_utils import cosine_similarity, get_embedding

from util import EMBEDDING_MODEL

def TRC(sentence: str, labels = ["This sentence indicates a possible foodborne illness incidient", 
                                 "This sentence doesn't indicate a possible foodborne illness incidient"]):
    label_embeddings = [get_embedding(label, engine=EMBEDDING_MODEL) for label in labels]

    def label_score(review_embedding, label_embeddings):
        return cosine_similarity(review_embedding, label_embeddings[1]) - cosine_similarity(review_embedding, label_embeddings[0])
    

    probas = label_score(get_embedding(sentence, engine=EMBEDDING_MODEL), label_embeddings)
    preds = "relevant" if probas > 0 else "irrelevant"

    return preds
