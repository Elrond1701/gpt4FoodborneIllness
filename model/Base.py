# from TRC import label_embeddings


class Base:
    def __init__(self):
        self.name = None
        self.model = None
    
    def train(self, x, y):
        self.model.fit(X=x, y=y)
    
    def inference(self, x):
        return self.model.predict(x)