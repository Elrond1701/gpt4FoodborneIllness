from sklearn.linear_model import LogisticRegressionCV
from model.Base import Base


class LogisticRegression(Base):

    def __init__(self):
        self.name = "LogisticRegression"
        self.model = LogisticRegressionCV(max_iter=1000)