# Load the dataset
from datasets import load_dataset
import pandas as pd



def main():
    train_ds = load_dataset('xsum','3.0.0', split='train[:3%]')
    val_ds = load_dataset('xsum', '3.0.0',split='train[3%:4%]')
    test_ds = load_dataset('xsum','3.0.0', split='train[4%:5%]')


    train_df = pd.DataFrame.from_dict(train_ds[:], orient='index').T[["document","summary"]] 
    val_df = pd.DataFrame.from_dict(val_ds[:], orient='index').T[["document","summary"]] 
    test_df = pd.DataFrame.from_dict(test_ds[:], orient='index').T[["document","summary"]] 

    train_df.to_csv("train.csv")
    test_df.to_csv("test.csv")
    val_df.to_csv("val.csv")



main()