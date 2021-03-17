import pandas as pd
import numpy as np
from feature_engineer.feature import Feature

class Last14(Feature):
    def __init__(self, regular_results, tourney_results, load=False):
        self.regular_results = regular_results
        self.tourney_results = tourney_results
        self.load = load

    def make_features(self):
        regular_results = self.regular_results

        last14days_stats_T1 = regular_results.loc[regular_results.DayNum>118].reset_index(drop=True)
        last14days_stats_T1['T1_TeamID'] =last14days_stats_T1['T1_TeamID'].astype(int)
        last14days_stats_T1['win'] = np.where(last14days_stats_T1['PointDiff']>0,1,0)
        last14days_stats_T1 = last14days_stats_T1.groupby(['Season','T1_TeamID'])['win'].mean().reset_index(name='T1_win_ratio_14d')

        last14days_stats_T2 = regular_results.loc[regular_results.DayNum>118].reset_index(drop=True)
        last14days_stats_T2['T2_TeamID'] =last14days_stats_T2['T2_TeamID'].astype(int)
        last14days_stats_T2['win'] = np.where(last14days_stats_T2['PointDiff']<0,1,0)
        last14days_stats_T2 = last14days_stats_T2.groupby(['Season','T2_TeamID'])['win'].mean().reset_index(name='T2_win_ratio_14d')

        self.feature1 = last14days_stats_T1
        self.feature2 = last14days_stats_T2

    def merge_features(self):    
        # Add winner's ordinals
        res = self.tourney_results.merge(self.feature1,on=['Season','T1_TeamID'], how='left')
        # Then add losser's ordinals
        res = res.merge(self.feature2,on=['Season','T2_TeamID'], how='left')

        return res        

