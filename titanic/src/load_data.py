import pandas as pd
import seaborn as sns

def load():
    data = sns.load_dataset("titanic")
    data.to_csv("data/titanic.csv", index=False)
    return data
