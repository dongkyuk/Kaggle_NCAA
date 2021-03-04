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

df = pd.read_csv(os.path.join(Config.DATA_DIR, 'submission_features.csv'))
submission_df = pd.read_csv(os.path.join(Config.DATA_DIR, 'MSampleSubmissionStage1.csv'))

features = ['T1_powerrank', 'T2_powerrank', 'adj_emT1', 'rankT1',
'adj_oT1', 'adj_dT1', 'luckT1', 'adj_emT2', 'rankT2', 'adj_oT2', 'adj_dT2', 'luckT2','OrdinalRankT1', 'OrdinalRankT2']

filename = "scaler.pkl"
scaler = pickle.load(open(os.path.join(Config.SAVE_DIR, filename), 'rb'))

for season in range(2015, 2020):
    df_test = df.loc[df['Season']==season]

    filename = f"{season}_model.pkl"
    model = pickle.load(open(os.path.join(Config.SAVE_DIR, filename), 'rb'))

    X_test = df_test[features]
    X_test = scaler.transform(X_test)

    df_test['Pred'] = model.predict_proba(X_test)[:, 1]
    df['Pred'].loc[df['Season']==season] = df_test['Pred']
    
print(df.head())
submission_df['Pred'] = df['Pred']
print(submission_df)
submission_df.to_csv('submit.csv', index=False)