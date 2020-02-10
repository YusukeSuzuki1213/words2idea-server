import csv
import pandas as pd
import sys

"""
csvの行をランダムに入れ替える

"""
def shuffle(path: str):
    df = pd.read_csv(path).sample(frac=1)
    df.to_csv('data/result.csv', header=False, index=False)

if __name__ == '__main__':
    path = sys.argv[1] 
    shuffle(path)