import ast
import random
import re
import numpy as np
import openai
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, TomekLinks, EditedNearestNeighbours, CondensedNearestNeighbour
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
from util import MODEL, SEED


def TRC_embedding(model_name, train_dat, test_dat, sampling_method="None", sample_size=None):
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
    samplers = {
        "None": None, 
        "SMOTEOver": SMOTE(sampling_strategy="auto", random_state=SEED),
        "RandomOver": RandomOverSampler(sampling_strategy="auto", random_state=SEED), 
        # "ADASYNOver": ADASYN(sampling_strategy="auto", random_state=SEED),
        "RandomUnder": RandomUnderSampler(sampling_strategy="auto", random_state=SEED), 
        "ClusterUnder": ClusterCentroids(sampling_strategy="auto", random_state=SEED), 
        "TomekUnder": TomekLinks(sampling_strategy="auto"), 
        "ENNUnder": EditedNearestNeighbours(sampling_strategy="auto"), 
        "CNNUnder": CondensedNearestNeighbour(sampling_strategy="auto", random_state=SEED)
    }
    sampler = samplers[sampling_method]
    if sampler is None:
        sampler_x, sampler_y = train_x, train_y
    else:
        sampler_x, sampler_y = sampler.fit_resample(train_x, train_y)
    if sample_size is None:
        pass
    else:
        types = train_dat["y"].unique()
        sample_lists = []
        for type in types:
            sample_list = np.where((train_dat["y"] == type))[0]
            sample_list = sample_list[random.sample(range(len(sample_list)), int(sample_size / len(types)))]
            sample_lists.extend(sample_list)
        sampler_x = sampler_x[sample_lists]
        sampler_y = sampler_y[sample_lists]
        print(len(sample_lists))
    
    test_x = np.array(list(test_dat['x'].apply(ast.literal_eval)))
    test_y = test_dat['y'].to_numpy()

    model.train(sampler_x, sampler_y)
    inference_y = model.inference(test_x)
    return confusion_matrix(test_y, inference_y), classification_report(test_y, inference_y), accuracy_score(test_y, inference_y)


def TRC_in_context(train_dat, test_dat, in_context_length=0):
    test_y = []
    inference_y = []
    for index, row in test_dat.iterrows():
        messages = "I'm an excellent linguist and an expert in foodborne illness. The task is to classfiy tweets into two classes. You can only answer it with \"yes\" or \"no\". The question and answer should be in the form of \r\n\r\n input: XXXX\r\noutput: yes/no\r\n\r\n"
        if in_context_length == 0:
            examples = ""
        else:
            examples = "Here is some examples.\r\n\r\n"
            # for i in range(in_context_length):
            #     examples += "input: " + + "\r\n" + "output: " + + "\r\n\r\n"
        messages += examples
        messages += "input: " + row["tweet"]
        messages += "\r\n"
        messages += "output: "
        response = openai.ChatCompletion.create(MODEL, messages)
        answer = response["choices"][0]["text"]
        answer = re.findall(r"output:(.*?)\r\n\r\n", answer, re.DOTALL)
        if answer:
            answer = answer[-1].strip()
        test_y.append(row["sentence_class"])
        inference_y.append(1 if answer else 0)
    return confusion_matrix(test_y, inference_y), classification_report(test_y, inference_y), accuracy_score(test_y, inference_y)