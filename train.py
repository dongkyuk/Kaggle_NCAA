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
from config import Config


df = pd.read_csv(os.path.join(Config.DATA_DIR, "train.csv"))

print(df['T1_powerrank'])

df['pred'] = 0


features = ['T1_powerrank', 'T2_powerrank', 'adj_emT1', 'rankT1', 'adj_oT1', 'adj_dT1', 'luckT1', 'adj_emT2', 'rankT2', 'adj_oT2', 'adj_dT2', 'luckT2', 'OrdinalRankT1', 'OrdinalRankT2', ]

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

    filename = f"{season}_model.pkl"
    with open(os.path.join(Config.SAVE_DIR, filename), 'wb') as file:
        pickle.dump(model, file)

    X_test = df_test[features]
    X_test = scaler.transform(X_test)

    df_test['pred'] = model.predict_proba(X_test)[:, 1]
    df['pred'].loc[df['Season']==season] = df_test['pred']

    # a = log_loss(df_test['win'],df_test['pred'])
    print(f"Season {season}, loss {log_loss(df_test['win'],df_test['pred'])}")

    # conservative = 0.0062
    # df_test['pred'].loc[df_test['pred']>0.5 + conservative] -= conservative
    # df_test['pred'].loc[df_test['pred']<0.5 - conservative] += conservative

    # # print(f"Season {season}, loss {log_loss(df_test['win'],df_test['pred'])}")
    # b = log_loss(df_test['win'],df_test['pred'])
    # print(a - b)

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(scaler.transform(df[features]), df['win'], test_size=0.33, random_state=42)

# model = LogisticRegression()
# model.fit(X_train,y_train)
# print(f"loss {log_loss(y_test,model.predict_proba(X_test)[:, 1])}")

# import matplotlib.pyplot as plt
print(f"Overall Loss {log_loss(df['win'],df['pred'])}")
# x = []
# y = []
# for i in range(60, 70):
#     conservative = i * 0.0001
#     df_2 = df.copy(deep=True)
#     df_2['pred'].loc[df_2['pred']>0.5 + conservative] -= conservative
#     df_2['pred'].loc[df_2['pred']<0.5 - conservative] += conservative
#     x.append(conservative)
#     y.append(log_loss(df['win'],df_2['pred']))
    
# plt.plot(x, y)
# plt.savefig('fig1.png', dpi=300)

#     # print(f"Overall Loss {log_loss(df['win'],df['pred'])}")

# # # df['pred'].loc[df['pred']>0.9] = 1
# # # df['pred'].loc[df['pred']<0.1] = 0

# # # df['pred'] = 0.5
# conservative = 0.0062
# df['pred'].loc[df['pred']>0.5 + conservative] -= conservative
# df['pred'].loc[df['pred']<0.5 - conservative] += conservative
# print(f"Overall Loss {log_loss(df['win'],df['pred'])}")

# df['ratingT1-2']= 100-4*np.log(df['T1_powerrank']+1)-df['T1_powerrank']/22
# df['ratingT2-2']= 100-4*np.log(df['T2_powerrank']+1)-df['T2_powerrank']/22
# df['pred'] = 1/(1+10**((df['ratingT2-2']-df['ratingT1-2'])/15))

# print(log_loss(df['win'],df['pred']))

# df['pred'] = 1/(1+10**((df['ratingT2']-df['ratingT1'])/15))

