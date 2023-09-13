from sklearn.tree import DecisionTreeClassifier
from model.Base import Base


class DecisionTree(Base):
    def __init__(self):
        self.name = "DecisionTree"
        self.model = DecisionTreeClassifier()