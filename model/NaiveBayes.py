from sklearn.naive_bayes import BernoulliNB
from model.Base import Base


class NaiveBayes(Base):

    def __init__(self):
        self.name = "NaiveBayes"
        self.model = BernoulliNB()