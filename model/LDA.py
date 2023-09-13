from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from model.Base import Base


class LDA(Base):

    def __init__(self):
        self.name = "LDA"
        self.model = LinearDiscriminantAnalysis()