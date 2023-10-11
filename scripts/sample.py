import pickle
import random
import numpy as np

import pandas as pd

from util import SEED


def sample(path, input_name, output_name, sampling_size, sampling_method):
    with open(path + input_name, 'rb') as file:
        dat = pickle.load(file)
    random.seed(SEED)

    train_x = dat['id'].to_numpy()
    train_y = dat['sentence_class'].to_numpy()
    # samplers = {
    #     "None": None, 
    #     "SMOTEOver": SMOTE(sampling_strategy="auto", random_state=SEED),
    #     "RandomOver": RandomOverSampler(sampling_strategy="auto", random_state=SEED), 
    # }
    # sampler = samplers[sampling_method]
    # if sampler is None:
    #     sampler_x, sampler_y = train_x, train_y
    # else:
    #     sampler_x, sampler_y = sampler.fit_resample(train_x, train_y)
    if sampling_size is None:
        pass
    else:
        types = dat["sentence_class"].unique()
        sample_lists = []
        for type in types:
            sample_list = np.where((dat["sentence_class"] == type))[0]
            sample_list = sample_list[random.sample(range(len(sample_list)), int(sampling_size / len(types)))]
            sample_lists.extend(sample_list)
        train_x = train_x[sample_lists]
        train_y = train_y[sample_lists]
    df = pd.DataFrame({"id": train_x})
    df.to_csv(path + output_name, index=True)
    