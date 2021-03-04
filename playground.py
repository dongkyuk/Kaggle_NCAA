import pandas as pd
import numpy as np
import warnings
import os
import pickle

from preprocess import preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from feature_engineer.logistic_team_rank import LogisticTeamRank
from feature_engineer.massey_ordinals import MasseyOrdinal
from feature_engineer.five_three_eight_rating import FiveThreeEight
from feature_engineer.kenpom import Kenpom

class Config():
    DATA_DIR = 'data'
    SAVE_DIR = 'save'

warnings.filterwarnings("ignore")
df = pd.read_csv(os.path.join(Config.DATA_DIR, "train.csv"))
print(df.loc[(df.T1_TeamID == 1107) & (df.Season == 2015)]['luckT1'])


