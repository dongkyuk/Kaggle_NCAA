import pandas as pd
from feature_engineer.feature import Feature

class SeasonStats(Feature):
    def __init__(self, regular_results, tourney_results, load=False):
        self.regular_results = regular_results
        self.tourney_results = tourney_results
        self.load = load

    def make_features(self):
        regular_results = self.regular_results
        boxscore_cols = [
            'T1_FGM', 'T1_FGA', 'T1_FGM3', 'T1_FGA3', 'T1_OR', 'T1_Ast', 'T1_TO', 'T1_Stl', 'T1_PF', 
                'T2_FGM', 'T2_FGA', 'T2_FGM3', 'T2_FGA3', 'T2_OR', 'T2_Ast', 'T2_TO', 'T2_Stl', 'T2_Blk',  
                'PointDiff'
        ]
        regular_results['PointDiff'] = regular_results['T1_Score'] - regular_results['T2_Score']
        print(regular_results)
        season_statistics = regular_results.groupby(["Season", 'T1_TeamID'])[boxscore_cols].mean().reset_index()
        season_statistics.columns = [''.join(col).strip() for col in season_statistics.columns.values]

        season_statistics_T1 = season_statistics.copy()
        season_statistics_T2 = season_statistics.copy()

        season_statistics_T1.columns = ["T1_" + x.replace("T1_","").replace("T2_","opponent_") for x in list(season_statistics_T1.columns)]
        season_statistics_T2.columns = ["T2_" + x.replace("T1_","").replace("T2_","opponent_") for x in list(season_statistics_T2.columns)]

        season_statistics_T1.columns.values[0] = "Season"
        season_statistics_T2.columns.values[0] = "Season"

        season_statistics_T1['T1_TeamID'] = season_statistics_T1['T1_TeamID'].astype(int)
        season_statistics_T2['T2_TeamID'] = season_statistics_T2['T2_TeamID'].astype(int)

        self.feature1 = season_statistics_T1
        self.feature2 = season_statistics_T2

    def merge_features(self):    
        print(self.feature1)
        # Add winner's ordinals
        res = self.tourney_results.merge(self.feature1,on=['Season','T1_TeamID'], how='left')
        # Then add losser's ordinals
        res = res.merge(self.feature2,on=['Season','T2_TeamID'], how='left')

        return res        