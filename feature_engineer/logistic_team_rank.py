import pandas as pd
import numpy as np
import statsmodels.api as sm
from feature_engineer.feature import Feature

class LogisticTeamRank(Feature):
    def __init__(self, tourney_results, seeds, regular_results, load=False):
        self.tourney_results = tourney_results
        self.seeds = seeds
        self.regular_results = regular_results
        self.feature_save_path = 'data/features/LogisticTeamRank.csv'
        self.load = load

    def team_quality(self, season):
        """
        Calculate team quality for each season seperately. 
        Team strength changes from season to season (students playing change!)
        So pooling everything would be bad approach!
        """
        formula = 'win~-1+T1_TeamID+T2_TeamID'
        glm = sm.GLM.from_formula(formula=formula, 
                                data=self.regular_results.loc[self.regular_results.Season==season,:], 
                                family=sm.families.Binomial()).fit()
        
        # extracting parameters from glm
        quality = pd.DataFrame(glm.params).reset_index()
        quality.columns = ['TeamID','beta']
        quality['Season'] = season
        # taking exp due to binomial model being used
        quality['quality'] = np.exp(quality['beta'])
        # only interested in glm parameters with T1_, as T2_ should be mirroring T1_ ones
        quality = quality.loc[quality.TeamID.str.contains('T1_')].reset_index(drop=True)
        quality['TeamID'] = quality['TeamID'].apply(lambda x: x[10:14]).astype(int)
        return quality
        

        return quality

    def make_features(self):
        all_seasons = [self.team_quality(season) for season in range(2010, 2021)]
        self.feature = pd.concat(all_seasons).reset_index(drop=True)
        self.feature.to_csv(self.feature_save_path, index=False)

    @staticmethod
    def merge_features_helper(tourney_results, team_quality, seeds):
        # Merge features
        team_quality_T1 = team_quality[['TeamID','Season','quality']]
        team_quality_T1.columns = ['T1_TeamID','Season','T1_quality']
        team_quality_T2 = team_quality[['TeamID','Season','quality']]
        team_quality_T2.columns = ['T2_TeamID','Season','T2_quality']
        
        tourney_results['T1_TeamID'] = tourney_results['T1_TeamID'].astype(int)
        tourney_results['T2_TeamID'] = tourney_results['T2_TeamID'].astype(int)
        tourney_results = tourney_results.merge(team_quality_T1, on = ['T1_TeamID','Season'], how = 'left')
        tourney_results = tourney_results.merge(team_quality_T2, on = ['T2_TeamID','Season'], how = 'left')

        # Add seed
        seeds['seed'] = seeds['Seed'].apply(lambda x: int(x[1:3]))
        seeds['division'] = seeds['Seed'].apply(lambda x: x[0])

        seeds_T1 = seeds[['Season','TeamID','seed','division']].copy()
        seeds_T2 = seeds[['Season','TeamID','seed','division']].copy()
        seeds_T1.columns = ['Season','T1_TeamID','T1_seed','T1_division']
        seeds_T2.columns = ['Season','T2_TeamID','T2_seed','T2_division']

        # Add power rank
        tourney_results = tourney_results.merge(seeds_T1, on = ['Season', 'T1_TeamID'], how = 'left')
        tourney_results = tourney_results.merge(seeds_T2, on = ['Season', 'T2_TeamID'], how = 'left')

        tourney_results['T1_powerrank'] = tourney_results.groupby(['Season','T1_division'])['T1_quality'].rank(method='dense', ascending=False).dropna(how='all').astype(int)
        tourney_results['T2_powerrank'] = tourney_results.groupby(['Season','T2_division'])['T2_quality'].rank(method='dense', ascending=False).dropna(how='all').astype(int)
        
        return tourney_results

    def merge_features(self):
        return LogisticTeamRank.merge_features_helper(self.tourney_results, self.feature, self.seeds)

