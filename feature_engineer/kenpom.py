import pandas as pd
from feature_engineer.feature import Feature

class Kenpom(Feature):
    def __init__(self, kenpom_df, tourney_results, load=False):
        self.kenpom_df = kenpom_df
        self.tourney_results = tourney_results
        self.feature_save_path = 'data/features/Kenpom.csv'
        self.load = load

    def make_features(self):
        # Get the last available data from each system previous to the tournament
        self.feature = self.kenpom_df[['Season','TeamID', 'adj_em', 'rank', 'adj_o','adj_d', 'luck']]
        self.feature.to_csv(self.feature_save_path, index=False)

    def merge_features(self):    
        # Add winner's ordinals
        res = self.tourney_results.merge(self.feature,left_on=['Season','T1_TeamID'],
                                right_on=['Season','TeamID']).drop(columns= ['TeamID'])

        # Then add losser's ordinals
        res = res.merge(self.feature,left_on=['Season','T2_TeamID'],
                                right_on=['Season','TeamID'],
                                suffixes = ['T1','T2']).drop(columns= ['TeamID'])
        return res        