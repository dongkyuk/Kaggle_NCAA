import pandas as pd
import os
from preprocess import preprocess
from feature_engineer.logistic_team_rank import LogisticTeamRank
from feature_engineer.massey_ordinals import MasseyOrdinal
from feature_engineer.five_three_eight_rating import FiveThreeEight
from feature_engineer.kenpom import Kenpom
from preprocess import preprocess
from config import Config

# Load Data
seeds = pd.read_csv(os.path.join(Config.DATA_DIR, 'MNCAATourneySeeds.csv'))
tourney_results = pd.read_csv(os.path.join(Config.DATA_DIR, 'MNCAATourneyCompactResults.csv'))
regular_results = pd.read_csv(os.path.join(Config.DATA_DIR, 'MRegularSeasonCompactResults.csv'))
ordinals_df = pd.read_csv(os.path.join(Config.DATA_DIR, 'MMasseyOrdinals.csv')).rename(columns={'RankingDayNum':'DayNum'})
rating_df = pd.read_csv(os.path.join(Config.DATA_DIR, '538ratingsMen.csv'))
kenpom_df = pd.read_csv(os.path.join(Config.DATA_DIR, 'NCAA2021_Kenpom.csv'))
submission_df = pd.read_csv(os.path.join(Config.DATA_DIR , "MSampleSubmissionStage1.csv"))

# Preprocess
submission_df['Season'] = submission_df['ID'].apply(lambda x: int(x.split('_')[0]))
submission_df['T1_TeamID'] = submission_df['ID'].apply(lambda x: int(x.split('_')[1]))
submission_df['T2_TeamID'] = submission_df['ID'].apply(lambda x: int(x.split('_')[2]))
train_df, regular_results = preprocess(tourney_results, regular_results)


# Add Features
train_df = LogisticTeamRank(train_df, seeds, regular_results).run()
train_df = MasseyOrdinal(ordinals_df, train_df, load=True).run()
train_df = Kenpom(kenpom_df, train_df, load=True).run()

submission_df = LogisticTeamRank(submission_df, seeds, regular_results, load=True).run()
submission_df = MasseyOrdinal(ordinals_df, submission_df, load=True).run()
submission_df = Kenpom(kenpom_df, submission_df, load=True).run()

# Save
train_df.to_csv(os.path.join(Config.DATA_DIR, 'train.csv'), index=False)
submission_df.to_csv(os.path.join(Config.DATA_DIR, 'submission_features.csv'), index=False)

