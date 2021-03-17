import pandas as pd
from feature_engineer.feature import Feature

class Seed(Feature):
    def __init__(self, seeds, tourney_results, load=False):
        self.seeds = seeds
        self.tourney_results = tourney_results
        self.feature_save_path = 'data/features/Seed.csv'
        self.load = load

    def make_features(self):
        # Add seed
        seeds = self.seeds
        seeds['seed'] = seeds['Seed'].apply(lambda x: int(x[1:3]))
        seeds['division'] = seeds['Seed'].apply(lambda x: x[0])
        self.feature = seeds[['Season','TeamID','seed','division']].copy()

    def merge_features(self):    
        # Add winner's ordinals
        res = self.tourney_results.merge(self.feature,left_on=['Season','T1_TeamID'],
                                right_on=['Season','TeamID']).drop(columns= ['TeamID'])
        # Then add losser's ordinals
        res = res.merge(self.feature,left_on=['Season','T2_TeamID'],
                                right_on=['Season','TeamID'],
                                suffixes = ['T1','T2']).drop(columns= ['TeamID'])

        res["seed_diff"] = res["seedT1"] - res["seedT2"]
        print(res)

        return res                
