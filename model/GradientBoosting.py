from sklearn.ensemble import GradientBoostingClassifier
from model.Base import Base


class GradientBoosting(Base):
    def __init__(self):
        self.name = "GradientBoosting"
        self.model = GradientBoostingClassifier()