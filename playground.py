# file1 = open('data/kenpom/kenpom_2010.txt', 'r')
# lines = file1.readlines() 
# new_lines = []
# for line in lines:
#     new_line = ""
#     sep = line.split()
#     if len(sep) == 0:
#         continue
#     if len(sep) != 21:
#         for i in range(len(sep)-21):
#             sep[1] += (" " + sep[2])
#             del sep[2]
#     print(len(sep))

#     for word in sep:
#         new_line += (word + "\t")
#     new_line += "\n"
#     new_lines.append(new_line)

# print(new_lines[0])

# # Writing to file
# file1 = open('data/kenpom/kenpom_2010_2.txt', 'w')
# file1.writelines(new_lines)
# file1.close()
import pandas as pd
import os
from preprocess import preprocess
from feature_engineer.logistic_team_rank import LogisticTeamRank
from feature_engineer.massey_ordinals import MasseyOrdinal
from feature_engineer.five_three_eight_rating import FiveThreeEight
from feature_engineer.kenpom import Kenpom
from preprocess import preprocess
from config import Config
import numpy as np
# Load Data
regular_results = pd.read_csv(os.path.join(Config.DATA_DIR, 'MRegularSeasonDetailedResults.csv'))
tourney_results = pd.read_csv(os.path.join(Config.DATA_DIR, 'MNCAATourneyDetailedResults.csv'))
seeds = pd.read_csv(os.path.join(Config.DATA_DIR, 'MNCAATourneySeeds.csv'))

tourney_results, regular_results = preprocess(tourney_results, regular_results)







# train_df = pd.read_csv(os.path.join(Config.DATA_DIR, 'train.csv'))

# # print(season_statistics.columns.tolist())
# # print(train_df.columns.tolist())

# train_df = train_df.merge(season_statistics_T1,on=['Season','T1_TeamID'], how='left')
# train_df = train_df.merge(season_statistics_T2,on=['Season','T2_TeamID'], how='left')

# train_df.to_csv(os.path.join(Config.DATA_DIR, 'train2.csv'), index=False)
