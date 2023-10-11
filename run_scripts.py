import argparse

import pandas as pd

from scripts.TRC import TRC_embedding, TRC_in_context
from scripts.sample import sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--name', default=None)
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--train_file', default=None)
    parser.add_argument('--test_file', default=None)
    args = parser.parse_args()

    if args.name is None:
        pass
    elif args.name == "TRC_embedding":
        train_dat = pd.read_csv(args.train_file)
        test_dat = pd.read_csv(args.test_file)
        print(TRC_embedding(model_name=args.model_name, train_dat=train_dat, test_dat=test_dat))
    elif args.name == "TRC_in_context":
        train_dat = pd.read_pickle(args.train_file)
        test_dat = pd.read_pickle(args.test_file)
        print(TRC_in_context(train_dat=train_dat, test_dat=test_dat))
    elif args.name == "sample":
        sampling_sizes = [50, 100, 200, 500, 1000, 2000]
        pathes = ["./data/English/LREC_BSC/", "./data/English/LREC_mv/", "./data/English/LREC_expert_label/"]
        for path in pathes:
            for sampling_size in sampling_sizes:
                sample(path=path, input_name="train.p", output_name=str(sampling_size) + ".csv", sampling_size=sampling_size, sampling_method="SMOTEOver")