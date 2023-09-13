import argparse

import pandas as pd

from scripts.TRC import TRC


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--name', default=None)
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--train_file', default=None)
    parser.add_argument('--test_file', default=None)
    args = parser.parse_args()

    if args.name is None:
        pass
    elif args.name == "TRC":
        train_dat = pd.read_csv(args.train_file)
        test_dat = pd.read_csv(args.test_file)
        print(TRC(model_name=args.model_name, train_dat=train_dat, test_dat=test_dat))