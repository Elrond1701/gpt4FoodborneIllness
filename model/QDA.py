from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from model.Base import Base


class QDA(Base):

    def __init__(self):
        self.name = "QDA"
        self.model = QuadraticDiscriminantAnalysis()