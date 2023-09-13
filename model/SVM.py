from sklearn.svm import SVC
from model.Base import Base


class SVM(Base):
    def __init__(self):
        self.name = "SVM"
        self.model = SVC()