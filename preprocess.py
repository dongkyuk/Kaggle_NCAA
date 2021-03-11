import pandas as pd
import numpy as np
from utils.time_function import time_function

@time_function
def preprocess(tourney_results, regular_results):
    def duplicate_swap(df):
        ''' Duplicate data by swapping team
        '''
        dfswap = df[['Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'WLoc', 'NumOT']]
        dfswap.loc[df['WLoc'] == 'H', 'WLoc'] = 'A'
        dfswap.loc[df['WLoc'] == 'A', 'WLoc'] = 'H'
        df.columns.values[6] = 'location'
        dfswap.columns.values[6] = 'location'         
        df.columns = [x.replace('W','T1_').replace('L','T2_') for x in list(df.columns)]
        dfswap.columns = [x.replace('L','T1_').replace('W','T2_') for x in list(dfswap.columns)]
        output = pd.concat([df, dfswap]).sort_index().reset_index(drop=True)
        
        return output

    # Make Swap Duplicates
    tourney_results = duplicate_swap(tourney_results)
    regular_results = duplicate_swap(regular_results)

    # Convert to str, so the model would treat TeamID them as factors
    regular_results['T1_TeamID'] = regular_results['T1_TeamID'].astype(str)
    regular_results['T2_TeamID'] = regular_results['T2_TeamID'].astype(str)

    # make it a binary task
    tourney_results['win'] = np.where(tourney_results['T1_Score']>tourney_results['T2_Score'], 1, 0)
    regular_results['win'] = np.where(regular_results['T1_Score']>regular_results['T2_Score'], 1, 0)

    return tourney_results, regular_results