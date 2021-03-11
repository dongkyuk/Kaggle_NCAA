import pandas as pd
import numpy as np
import warnings
import os
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from config import Config
from preprocess import preprocess
from feature_engineer.logistic_team_rank import LogisticTeamRank
from feature_engineer.massey_ordinals import MasseyOrdinal
from feature_engineer.five_three_eight_rating import FiveThreeEight
from feature_engineer.kenpom import Kenpom


df = pd.read_csv(os.path.join(Config.DATA_DIR, 'submission_features.csv'))
submission_df = pd.read_csv(os.path.join(Config.DATA_DIR, 'MSampleSubmissionStage1.csv'))

# features = ['adj_emT1', 'rankT1',
#  'adj_oT1', 'adj_dT1', 'luckT1', 'adj_emT2', 'rankT2', 'adj_oT2', 'adj_dT2', 'luckT2']

features =['T1_powerrank', 'T2_powerrank', 'adj_emT1', 'rankT1',
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


# conservative = 0.0062
# df['Pred'].loc[df['Pred']>0.5 + conservative] -= conservative
# df['Pred'].loc[df['Pred']<0.5 - conservative] += conservative

print(df.head())
print(submission_df)

submission_df = submission_df.drop(['Pred'], axis = 1)
submission_df = submission_df.merge(df, on = ['ID'], how = 'left')
submission_df = submission_df[['ID', 'Pred']]

print(submission_df)
submission_df.to_csv('submit.csv', index=False)
