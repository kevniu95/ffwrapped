from enum import Enum
import datetime
import pathlib
import os
import re
import pandas as pd

class ScoringType(Enum):
    PPR = 1
    HPPR = 0.5
    NPPR = 0

    def points_name(self) -> str:
        return 'Pts_' + self.name
    
    def lower_name(self) -> str:
        return self.name.lower()
    
    def adp_column_name(self) -> str:
        if self.name == 'PPR':
            return 'AverageDraftPositionPPR'
        elif self.name == 'NPPR':
            return 'AverageDraftPosition'
        elif self.name == 'HPPR':
            return 'AverageDraftPositionHPPR'
        else:
            return None

def thisFootballYear() -> int:
    # If current date is between August and December, use this year
    # Otherwise, round down year down
    year = datetime.datetime.now().year
    if 8 <= datetime.datetime.now().month <= 12:
        return year
    else:
        return year - 1

WEEKLY_STATS_PARAMS = {'request' : '1',
                                    'draft_pick_type' : 'overall',
                                    'player_game_num_career_max' : '400',
                                    'comp_type' : 'reg',
                                    'order_by' : 'fantasy_points',
                                    'season_start' : '1',
                                    'team_game_num_season_max' : '17',
                                    'team_game_num_season_min' : '1',
                                    'weight_max' : '500',
                                    'week_num_season_max' : '22',
                                    'rookie' : 'N',
                                    'timeframe': 'seasons',
                                    'player_game_num_season_max' : '18',
                                    'year_min' : '2021',
                                    'qb_start_num_career_min' : '1',
                                    'match' : 'player_game',
                                    'year_max' : '2021',
                                    'player_game_num_season_min' : '1',
                                    'season_end=' : '1',
                                    'qb_start_num_career_max' : '400',
                                    'week_num_season_min' : '1',
                                    'player_game_num_career_min' : '1',
                                    'cstat[1]' : 'targets',
                                    'ccomp[1]' : 'gt',
                                    'cval[1]' : '0',
                                    'cstat[2]' : 'pass_att',
                                    'ccomp[2]' : 'gt',
                                    'cval[2]' : '0',
                                    'cstat[3]' : 'rush_att',
                                    'ccomp[3]' : 'gt',
                                    'cval[3]' : '0',
                                    'cstat[4]' : 'fgm',
                                    'ccomp[4]' : 'gt',
                                    'cval[4]' : '0',
                                    'offset' : '0'}

# Load regression results
def loadDatasetAfterBaseRegression(df_path : str = None
                                   #, use_compressed : bool = True
                                   ) -> pd.DataFrame:
    path = pathlib.Path(__file__).parent.resolve()
    os.chdir(path)
    if not df_path:
        df_path = '../../data/regression/reg_w_preds_1.p'
        # if use_compressed:
            # df_path = (re.sub(r'\.p$', '_compressed.p', df_path))        
    return pd.read_pickle(df_path)