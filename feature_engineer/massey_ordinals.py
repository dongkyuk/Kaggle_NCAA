import pandas as pd
from feature_engineer.feature import Feature

class MasseyOrdinal(Feature):
    def __init__(self, ordinals_df, tourney_results, ref_system = 'POM'):
        self.ordinals_df = ordinals_df
        self.tourney_results = tourney_results
        self.ref_system = ref_system

    def make_features(self):
        # Get the last available data from each system previous to the tournament
        self.feature = self.ordinals_df.groupby(['SystemName','Season','TeamID']).last().reset_index().drop(columns='DayNum')    
        self.feature = self.feature.loc[self.feature.SystemName==self.ref_system]

    def merge_features(self):    
        # Add winner's ordinals
        res = self.tourney_results.merge(self.feature,left_on=['Season','T1_TeamID'],
                                right_on=['Season','TeamID']).drop(columns= ['TeamID'])

        # Then add losser's ordinals
        res = res.merge(self.feature,left_on=['Season','T2_TeamID','SystemName'],
                                right_on=['Season','TeamID','SystemName'],
                                suffixes = ['T1','T2']).drop(columns= ['TeamID'])
        return res        