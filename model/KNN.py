from sklearn.neighbors import KNeighborsClassifier
from model.Base import Base


class KNN(Base):

    def __init__(self):
        self.name = "KNN"
        self.model = KNeighborsClassifier()