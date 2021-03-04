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

class Config():
    DATA_DIR = 'data'
    SAVE_DIR = 'save'

warnings.filterwarnings("ignore")

df = pd.read_csv(os.path.join(Config.DATA_DIR, "train.csv"))
df['pred'] = 0

features = ['T1_powerrank', 'T2_powerrank', 'adj_emT1', 'rankT1',
 'adj_oT1', 'adj_dT1', 'luckT1', 'adj_emT2', 'rankT2', 'adj_oT2', 'adj_dT2', 'luckT2','OrdinalRankT1', 'OrdinalRankT2']

scaler = StandardScaler()
scaler.fit(df[features])
filename = "scaler.pkl"
with open(os.path.join(Config.SAVE_DIR, filename), 'wb') as file:
    pickle.dump(scaler, file)

for season in range(2003, 2020):
    df_test = df.loc[df['Season']==season]
    df_train = df.loc[df['Season']!=season]

    X = df_train[features]
    X = scaler.transform(X)
    y = df_train[['win']]

    model = LogisticRegression()
    model.fit(X, y)

    X_test = df_test[features]
    X_test = scaler.transform(X_test)

    df_test['pred'] = model.predict_proba(X_test)[:, 1]
    df['pred'].loc[df['Season']==season] = df_test['pred']
    
    filename = f"{season}_model.pkl"
    with open(os.path.join(Config.SAVE_DIR, filename), 'wb') as file:
        pickle.dump(model, file)

    print(f"Season {season}, loss {log_loss(df_test['win'],df_test['pred'])}")



print(f"Overall Loss {log_loss(df['win'],df['pred'])}")

conservative = 0.01
df['pred'].loc[df['pred']>0.5 + conservative] -= conservative
df['pred'].loc[df['pred']<0.5 - conservative] += conservative

# # df['pred'].loc[df['pred']>0.9] = 1
# # df['pred'].loc[df['pred']<0.1] = 0

# # df['pred'] = 0.5
print(f"Overall Loss {log_loss(df['win'],df['pred'])}")

# df['ratingT1-2']= 100-4*np.log(df['T1_powerrank']+1)-df['T1_powerrank']/22
# df['ratingT2-2']= 100-4*np.log(df['T2_powerrank']+1)-df['T2_powerrank']/22
# df['pred'] = 1/(1+10**((df['ratingT2-2']-df['ratingT1-2'])/15))

# print(log_loss(df['win'],df['pred']))

# df['pred'] = 1/(1+10**((df['ratingT2']-df['ratingT1'])/15))

