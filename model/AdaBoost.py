from sklearn.ensemble import AdaBoostClassifier
from model.Base import Base


class AdaBoost(Base):
    def __init__(self):
        self.name = "AdaBoost"
        self.model = AdaBoostClassifier()