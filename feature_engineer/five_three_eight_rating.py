import pandas as pd
from feature_engineer.feature import Feature

class FiveThreeEight(Feature):
    def __init__(self, rating_df, tourney_results):
        self.rating_df = rating_df
        self.tourney_results = tourney_results

    def make_features(self):
        # Get the last available data from each system previous to the tournament
        self.feature = self.rating_df.drop(columns=['TeamName'])

    def merge_features(self):    
        # Add winner's ordinals
        res = self.tourney_results.merge(self.feature,left_on=['Season','T1_TeamID'],
                                right_on=['Season','TeamID']).drop(columns= ['TeamID'])

        # Then add losser's ordinals
        res = res.merge(self.feature,left_on=['Season','T2_TeamID'],
                                right_on=['Season','TeamID'],
                                suffixes = ['T1','T2']).drop(columns= ['TeamID'])
        return res        