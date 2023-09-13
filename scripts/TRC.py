import ast
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from model.AdaBoost import AdaBoost
from model.DecisionTree import DecisionTree
from model.GaussianProcess import GaussianProcess
from model.GradientBoosting import GradientBoosting
from model.KNN import KNN
from model.LDA import LDA
from model.LogisticRegression import LogisticRegression
from model.NaiveBayes import NaiveBayes
from model.QDA import QDA
from model.RandomForest import RandomForest
from model.SVM import SVM
from util import string_to_numpy_array


def TRC(model_name, train_dat, test_dat):
    models = {
        "Adaboost": AdaBoost(), 
        "DecisionTree": DecisionTree(), 
        "GaussianProcess": GaussianProcess(), 
        "GradientBoosting": GradientBoosting(), 
        "KNN": KNN(), 
        "LDA": LDA(), 
        "LogisticRegression": LogisticRegression(), 
        "NaiveBayes": NaiveBayes(), 
        "QDA": QDA(), 
        "RandomForest": RandomForest(), 
        "SVM": SVM()
    }
    model = models[model_name]

    train_x = np.array(list(train_dat['x'].apply(ast.literal_eval)))
    train_y = train_dat['y'].to_numpy()
    test_x = np.array(list(test_dat['x'].apply(ast.literal_eval)))
    test_y = test_dat['y'].to_numpy()

    model.train(train_x, train_y)
    inference_y = model.inference(test_x)
    return confusion_matrix(test_y, inference_y), classification_report(test_y, inference_y)