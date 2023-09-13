from sklearn.ensemble import RandomForestClassifier
from model.Base import Base


class RandomForest(Base):
    def __init__(self):
        self.name = "RandomForest"
        self.model = RandomForestClassifier()