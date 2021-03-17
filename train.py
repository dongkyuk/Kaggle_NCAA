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
import optuna.integration.lightgbm as lgb
from lightgbm import LGBMClassifier
from flaml import AutoML
from xgboost import XGBClassifier, plot_importance
# automl = AutoML()
# automl.fit(X_train, y_train, task="classification")

df = pd.read_csv(os.path.join(Config.DATA_DIR, "train.csv"))
df['pred'] = 0
features = ['seedT1', 'seedT2', 'seed_diff', 
'T1_quality', 'T2_quality', 'T1_powerrank', 'T2_powerrank', 'OrdinalRankT1', 
'OrdinalRankT2', 'adj_emT1', 'rankT1', 'adj_oT1', 'adj_dT1', 'luckT1', 
'adj_emT2', 'rankT2', 'adj_oT2', 'adj_dT2', 'luckT2',
'T1_FGM', 'T1_FGA', 'T1_FGM3', 'T1_FGA3', 
'T1_OR', 'T1_Ast', 'T1_TO', 'T1_Stl', 'T1_PF', 'T2_FGM', 'T2_FGA', 'T2_FGM3', 'T2_FGA3', 'T2_OR', 'T2_Ast', 'T2_TO', 'T2_Stl', 'T2_PF', 
'T1_win_ratio_14d', 'T2_win_ratio_14d']

print(df.columns.tolist())
print(df[features].info())


scaler = StandardScaler()
scaler.fit(df[features])

filename = "scaler.pkl"
with open(os.path.join(Config.SAVE_DIR, filename), 'wb') as file:
    pickle.dump(scaler, file)

# df = df[df['Season'] > 2010]

# model = AutoML()
# X_train = df[features]
# y_train = df[['win']].to_numpy()
# # def cauchyobj(preds, dtrain):
# #     labels = dtrain.get_label()
# #     c = 5000 
# #     x =  preds-labels    
# #     grad = x / (x**2/c**2+1)
# #     hess = -c**2*(x**2-c**2)/(x**2+c**2)**2
# #     return grad, hess
# model.fit(X_train, y_train,eval_method='cv', estimator_list=['xgboost'])
# model = XGBClassifier(eval_metric =  'mae', booster = 'gbtree', eta = 0.05,
#     subsample = 0.35,
#     colsample_bytree = 0.7,
#     num_parallel_tree = 3, #recommend 10
#     min_child_weight = 40,
#     gamma = 10,
#     max_depth =  3,
#     silent = 1,
#     obj=cauchyobj)
# model.fit(X_train, y_train)
# plot_importance(model)
# import matplotlib.pyplot as plt

# plt.savefig('fig2.png', dpi=300)

def cauchyobj(preds, dtrain):
    labels = dtrain.get_label()
    c = 5000 
    x =  preds-labels    
    grad = x / (x**2/c**2+1)
    hess = -c**2*(x**2-c**2)/(x**2+c**2)**2
    return grad, hess


for season in range(2007, 2020):
    if season == 2008:
        continue
    df_test = df.loc[df['Season']==season]
    df_train = df.loc[df['Season']!=season]

    X_train = df_train[features]
    X_train = scaler.transform(X_train)
    y_train = df_train[['win']].to_numpy()
    # model = LogisticRegression()
    # model = LogisticRegression(C=0.2528171435570508, n_jobs=-1, penalty='l1', solver='saga')
    # model = XGBClassifier(eval_metric =  'mae', booster = 'gbtree', eta = 0.05,
    #     subsample = 0.35,
    #     colsample_bytree = 0.7,
    #     num_parallel_tree = 3, #recommend 10
    #     min_child_weight = 40,
    #     gamma = 10,
    #     max_depth =  3,
    #     silent = 1,
    #     obj=cauchyobj)
    model = XGBClassifier(base_score=0.5, booster='gbtree',
              colsample_bylevel=0.8428363996690276, colsample_bynode=1,
              colsample_bytree=0.9955681869168213, gamma=0, gpu_id=-1,
              grow_policy='lossguide', importance_type='gain',
              interaction_constraints='', learning_rate=0.05366186508698025,
              max_delta_step=0, max_depth=0, max_leaves=57,
              min_child_weight=12.943839270411829,
              monotone_constraints='()', n_estimators=64, n_jobs=-1,
              num_parallel_tree=1, random_state=0,
              reg_alpha=1.7293875695646644e-06, reg_lambda=0.03780140213466081,
              scale_pos_weight=1, subsample=1.0, tree_method='hist',
              validate_parameters=1, verbosity=0, obj=cauchyobj)
    # model = LogisticRegression(C=0.25, n_jobs=-1, penalty='l1',
    #                solver='saga')
    # model = LogisticRegression(C=7.9999999999999964, n_jobs=-1, penalty='l1', solver='saga')
    # model = LGBMClassifier(colsample_bytree=0.8895927710640621,
    #            learning_rate=0.038101034402569234, max_bin=511,
    #            min_child_weight=0.27238244872515077, n_estimators=181,
    #            num_leaves=8, objective='binary',
    #            reg_alpha=1.412452673922427e-06, reg_lambda=0.3293813298875299,
    #            subsample=0.9120215384726259)
    model.fit(X_train, y_train)

    X_test = df_test[features]
    X_test = scaler.transform(X_test)
    y_test = df_test[['win']].to_numpy()

    # X_train = df[features]
    # y_train = df[['win']].to_numpy()
    # model = AutoML()
    # # model.fit(X_train = X_train, y_train = y_train, X_val=X_test, y_val=y_test, task='classification', estimator_list=['lrl1'])
    # model.fit(X_train, y_train,eval_method='cv')
    # print(automl.model)
    filename = f"{season}_model.pkl"
    with open(os.path.join(Config.SAVE_DIR, filename), 'wb') as file:
        pickle.dump(model, file)

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

import matplotlib.pyplot as plt
df = df.loc[df['Season']!=2010]
# print(df)
print(f"Overall Loss {log_loss(df['win'],df['pred'])}")
from sklearn.linear_model import LinearRegression
# print(df['pred'])

model = LinearRegression()
x = df['pred'].to_numpy().reshape(-1, 1)
model.fit(x, df['win'])
df['pred'] = model.predict(x)
# print(df['pred'])
print(f"Overall Loss {log_loss(df['win'],df['pred'])}")


x = []
y = []
for i in range(0, 200):
    conservative = i * 0.0001
    df_2 = df.copy(deep=True)
    df_2['pred'].loc[df_2['pred']>0.5 + conservative] -= conservative
    df_2['pred'].loc[df_2['pred']<0.5 - conservative] += conservative
    x.append(conservative)
    y.append(log_loss(df['win'],df_2['pred']))
    
plt.plot(x, y)
plt.savefig('fig1.png', dpi=300)


#     # print(f"Overall Loss {log_loss(df['win'],df['pred'])}")

# # # df['pred'].loc[df['pred']>0.9] = 1
# # # df['pred'].loc[df['pred']<0.1] = 0

# # # df['pred'] = 0.5
# conservative = 0.0075
# df['pred'].loc[df['pred']>0.5 + conservative] -= conservative
# df['pred'].loc[df['pred']<0.5 - conservative] += conservative
# print(f"Overall Loss {log_loss(df['win'],df['pred'])}")

# df['ratingT1-2']= 100-4*np.log(df['T1_powerrank']+1)-df['T1_powerrank']/22
# df['ratingT2-2']= 100-4*np.log(df['T2_powerrank']+1)-df['T2_powerrank']/22
# df['pred'] = 1/(1+10**((df['ratingT2-2']-df['ratingT1-2'])/15))

# print(log_loss(df['win'],df['pred']))

# df['pred'] = 1/(1+10**((df['ratingT2']-df['ratingT1'])/15))

