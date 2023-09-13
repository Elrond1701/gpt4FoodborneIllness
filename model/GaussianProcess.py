from sklearn.gaussian_process import GaussianProcessClassifier
from model.Base import Base


class GaussianProcess(Base):

    def __init__(self):
        self.name = "GaussianProcess"
        self.model = GaussianProcessClassifier()