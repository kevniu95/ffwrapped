import random
from abc import ABC
from typing import List, Dict, Set, Any
import logging
import pandas as pd
import numpy as np
import sklearn
from sklearn import pipeline

from ..domain.common import ScoringType, loadDatasetAfterBaseRegression

from ..util.logger_config import setup_logger
LOG_LEVEL = logging.INFO
logger = setup_logger(__name__, level = logging.INFO)


'''
Simulate fantasy football draft
'''

ROSTER_CONFIG_DEFAULT = {'QB' : 1,
                        'RB' : 2,
                        'WR' : 2,
                        'FLEX' : 1,
                        'TE' : 1,
                        'D/ST' : 1,
                        'K' : 1,
                        'BENCH' : 7}
    
class Player():
    def __init__(self, id, name, position):
        self.id : str = id
        self.name : str = name
        self.position : str = position
    
    def __str__(self):
        return self.name + ': ' + self.id

    def __repr__(self):
        return self.name + ': ' + self.id


class DraftPlayer(Player):
    def __init__(self, id, name, position):
        super().__init__(id, name, position)
        self.drafted : bool = False
    
    def wasDrafted(self) -> bool:
        if self.drafted:
            print("Player was drafted")
        return self.drafted

class ADPDraftPlayer(DraftPlayer):
    def __init__(self, id, name, position, adp):
        super().__init__(id, name, position)
        self.adp = adp

class PlayerPool(ABC):
    def __init__(self, available, unavailable):
        pass
        
class ADPPlayerPool(PlayerPool):
    '''
    Player pool implemented as pandas dataframe with ADP
    '''
    def __init__(self, df : pd.DataFrame, scoringType : ScoringType):
        self.df = df
        self.scoringType = scoringType
            
class Team():
    def __init__(self, id : str, rosterConfig : Dict[str, int] = ROSTER_CONFIG_DEFAULT):
        self.id = id
        self.rosterConfig = rosterConfig
        self.size : int = sum([v for v in rosterConfig.values()])
        self.roster : Dict[str, List[DraftPlayer]] = {k : [] for k in self.rosterConfig.keys()}
    
    @property
    def rosterSize(self) -> int:
        return sum([len(v) for v in self.roster.values()])

    @property
    def unfilledStarters(self) -> Set[str]:
        pos_set = {'QB', 'RB', 'WR', 'TE', 'FLEX'}
        return {i for i in pos_set if self._hasRoomAtPosition(i)}

    def _completeRoster(self) -> bool:
        if self.rosterSize >= self.size:
            print("Roster is complete!")
            return True
        return False

    def _hasRoomAtPosition(self, pos : str) -> bool:
        return len(self.roster[pos]) < self.rosterConfig[pos]
    
    def _addPlayer(self, player : DraftPlayer, pos : str) -> None:
        self.roster[pos].append(player)
        
    def addPlayer(self, player : DraftPlayer) -> bool:
        if self._completeRoster() or player.wasDrafted():
            return False
        
        pos = player.position
        positions_to_check = [pos, 'FLEX', 'BENCH'] if pos in ['RB', 'TE', 'WR'] else [pos, 'BENCH']
        for position in positions_to_check:
            if self._hasRoomAtPosition(position):
                self._addPlayer(player, position)
                return True

        print(f"You cannot select {player.name}")
        return False
    
    def removePlayer(self, id : str, pos : str) -> bool:
        # Assumes removed player is last player picked at his position
        if self.rosterSize == 0:
            return False
        elif id in [i.id for i in self.roster[pos]]:
            self.roster[pos] = self.roster[pos][:-1]
            print(self.roster)
            return True
        elif id in [i.id for i in self.roster['BENCH']]:
            self.roster['BENCH'] = self.roster[pos][:-1]
            print(self.roster)
            return True
        else:
            print("Couldn't find player on roster!")
            return False
    
    def selectFromPool(self, pool : ADPPlayerPool, pickNum : int, temperature : float, numTeams: int = 10) -> str:
        scoringType = pool.scoringType
        positionsNeeded : Set[str] = self.unfilledStarters
        pool_sub = pool.df[(pool.df['team'].isnull())].copy()
        
        teamPickNum = pickNum // numTeams + 1
        if len(positionsNeeded) == 0:
            pass
        elif 8 < (teamPickNum) < 12:
            # Force prioritization of still undrafted positions
            for i in positionsNeeded:
                pool_sub.loc[pool_sub['FantPos'] == i, scoringType.adp_column_name()] -= 2
        elif teamPickNum > 11 and len(positionsNeeded) > 0:
            # Force flex selection if not done by certain round
            if 'FLEX' in positionsNeeded:
                positionsNeeded.update(['RB','TE','WR'])
            pool_sub = pool_sub.loc[pool_sub['FantPos'].isin(positionsNeeded)]
        
        if len(pool_sub) == 0:
            print(positionsNeeded)

        pool_sub = self._prepDfProbs(pool_sub, temperature, scoringType)
        selectedId = np.random.choice(pool_sub['pfref_id'], p = pool_sub['probabilities'])
        return selectedId
    
    def finalizeSelection(self, selectedId : str, pool : ADPPlayerPool, pickNum : int) -> None:
        selectedRow = pool.df.loc[pool.df['pfref_id'] == selectedId]
        player = DraftPlayer(selectedRow['pfref_id'].item(), selectedRow['Player'].item(), selectedRow['FantPos'].item())
        if self.addPlayer(player):
            pool.df.loc[pool.df['pfref_id'] == selectedId, ['team', 'pick']] = [self.id, pickNum]
    
    def _getRosterConfigOnePosition(self, pool_df : pd.DataFrame, position : str) -> Dict[str, Any]:
        """
        1. Get summary stats for roster config at one position
        2. These are later used as inputs in model based on roster composition of each team
        """
        df = pool_df.copy()
        teamId = self.id
        if position == 'FLEX':
            temp_df = df[(df['team'] == teamId) & (df['FantPos'].isin(['RB','WR','TE']))].copy()
            temp_df.sort_values(['pred'], ascending = False, inplace = True)
            top_rbs = temp_df[temp_df['FantPos'] == 'RB'].head(2).index
            top_wrs = temp_df[temp_df['FantPos'] == 'WR'].head(2).index
            top_te = temp_df[temp_df['FantPos'] == 'TE'].head(1).index
            temp_df.drop(index=top_rbs, inplace=True)
            temp_df.drop(index=top_wrs, inplace=True)
            temp_df.drop(index=top_te, inplace=True)
            n_starters = 2
        else:
            temp_df = df[(df['team'] == teamId) & (df['FantPos'] == position)].copy()
            n_starters = 2 if position in ['RB', 'WR','FLEX'] else 1
        temp_df.reset_index(drop=True, inplace=True)
        
        top_players = temp_df.head(n_starters)
        
        num_others = len(temp_df) - n_starters
        avg_pred_points_other = temp_df.iloc[n_starters:]['pred'].mean()
        avg_std_error_other = temp_df.iloc[n_starters:]['var_pred'].mean()
        
        row = {
            'team': teamId,
            'FantPos': position,
            'A_pred_points': top_players.loc[0, 'pred'] if 0 in top_players.index else 0,
            'A_std_error': top_players.loc[0, 'var_pred'] if 0 in top_players.index else 10000,
            'B_pred_points': top_players.loc[1, 'pred'] if 1 in top_players.index else 0,
            'B_std_error': top_players.loc[1, 'var_pred'] if 1 in top_players.index else 10000,
            'num_others': num_others,
            'avg_pred_points_other': avg_pred_points_other if avg_pred_points_other is not np.nan else 0,
            'avg_std_error_other': avg_std_error_other if avg_std_error_other is not np.nan else 10000,
        }
        return row
    
    def getRosterConfigOneTeam(self, pool_df : pd.DataFrame, position : str = None) -> List[Dict[str, Any]]:
        end_list = []
        if position is None:
            pos_list = ['QB', 'RB', 'WR', 'TE', 'FLEX']
        else:
            pos_list = [position]
        for pos in pos_list:
            row = self._getRosterConfigOnePosition(pool_df, pos)
            end_list.append(row)
        return end_list
        
    def recalculate_values(self, 
                           pool_df : pd.DataFrame, 
                           models : Dict[str, sklearn.pipeline.Pipeline],
                           remaining_picks : int,
                           position : str = None)-> pd.DataFrame:
        if len(remaining_picks) < 2:
            return pool_df
        this_pick = remaining_picks[0]
        next_pick = remaining_picks[1]
        pos_list = [position]
        if position is None:
            pos_list = ['QB', 'RB', 'WR', 'TE', 'FLEX']
        for pos in pos_list:
            base_vars = ['A_pred_points',
                'A_std_error',
                'num_others',
                'avg_pred_points_other',
                'avg_std_error_other'
                ]
            model = models[pos]
            if pos in['RB','WR']:
                base_vars = base_vars + ['B_pred_points', 'B_std_error']

            # Get current score at position
            pos_row = self._getRosterConfigOnePosition(pool_df, position)
            df_extended = pd.DataFrame([pos_row], columns=pos_row.keys())
            base = model.predict(df_extended[base_vars])
            
            for num, row in pool_df.loc[(pool_df['FantPos'] == pos) & (pool_df['team'].isnull())].iterrows():
                pool_df.loc[pool_df['pfref_id'] == row['pfref_id'], 'team'] = self.id
                summary_row = self._getRosterConfigOnePosition(pool_df, pos)
                df_extended_new = pd.DataFrame([summary_row], columns=summary_row.keys())
                a = model.predict(df_extended_new[base_vars])
                pool_df.loc[pool_df['pfref_id'] == row['pfref_id'], 'team'] = np.nan
                pool_df.loc[pool_df['pfref_id'] == row['pfref_id'], 'valueAdd'] = a - base
        
        sub = pool_df[pool_df['team'].isnull()].reset_index().drop('index', axis = 1)
        picks_to_wait = next_pick - this_pick
        remainers = sub.iloc[picks_to_wait:]
        top_n = remainers.groupby('FantPos', group_keys=False).apply(lambda x: x.head(3)).reset_index(drop=True)
        base = top_n[['FantPos','valueAdd','pred']].groupby('FantPos').mean().reset_index()
        base.rename(columns = {'valueAdd': "valueAdd_base", 'pred' : 'pred_base'}, inplace = True)
        
        if 'valueAdd_base' in pool_df.columns:
            print("Dropping columns...")
            pool_df.drop(['valueAdd_base', 'pred_base'], axis = 1, inplace = True)
        pool_df = pool_df.merge(base, on ='FantPos', how = 'left')
        pool_df['valueAdd_diff'] = pool_df['valueAdd'] - pool_df['valueAdd_base']
        pool_df['pred_diff'] = pool_df['pred'] - pool_df['pred_base']
        return pool_df
        
    def _prepDfProbs(self, df : pd.DataFrame, temperature : float, scoringType : ScoringType) -> pd.DataFrame:
        MAX_ADP = 300
        inverted_adp = MAX_ADP + 1 - df[scoringType.adp_column_name()]
        probabilities = softmax(inverted_adp, temperature)
        df['probabilities'] = probabilities
        return df

    def __str__(self) -> str:
        return self.id
    
    def __repr__(self) -> str:
        return self.id
    
class League():
    def __init__(self, numTeams : int, year: int ,rosterConfig : Dict[str, int] = None):
        self.numTeams = numTeams
        self.rosterConfig = rosterConfig
        self.teams : Set[Team] = set([])
        self.year = year
        
        if not self.numTeams:
            self.numTeams = 10
        if not self.rosterConfig:
            self.rosterConfig = ROSTER_CONFIG_DEFAULT
    
    def initTeams(self):
        # Will implement
        pass

    def initTeamsDefault(self):
        self.teams = [Team('team_' + str(i + 1), self.rosterConfig) for i in range(self.numTeams)]
    
    def getTeamFromId(self, id : str) -> Team:
        team = [i for i in self.teams if i.id == id]
        if len(team) == 0:
            return None
        return team[0]

class Draft():
    def __init__(self, pool : PlayerPool, league : League, year: int):
        self.pool = pool
        self.league = league
        self.year = year
        self.rounds = sum([v for v in self.league.rosterConfig.values()]) - 2

class SnakeDraft(Draft):
    def __init__(self, pool : PlayerPool, league : League, year: int):
        super().__init__(pool, league, year)
        self.draftOrder = None
    
    def getDraftOrder(self, shuffle : bool = True) -> List[Team]:
        finalList = []
        teamList = list(self.league.teams)
        if shuffle:
            random.shuffle(teamList)
        if self.rounds % 2 == 1:
            raise Exception(f"League configured with {self.rounds} odd number of roster spots, won't do snake draft")
        for i in range(self.rounds // 2):
            finalList.extend(teamList)
            finalList.extend(teamList[::-1])
        self.draftOrder = finalList
        return finalList

def softmax(x : pd.Series, T : float = 1.0) -> pd.Series:
    x = x / T 
    e_x = np.exp(x - np.max(x))  # prevent overflow with subtraction
    return e_x / e_x.sum(axis=0)

def initPlayerPoolDfFromRegDataset(year: int, scoringType : ScoringType, colSubset : List[str], use_compressed : bool = False) -> pd.DataFrame:
    reg_df = loadDatasetAfterBaseRegression(use_compressed=use_compressed)
    reg_df = reg_df.loc[(reg_df ['Year'] == year) 
                    & (reg_df ['foundAdp'].isin(['left_only', 'both'])),
                    colSubset
                    ].copy()
    reg_df.sort_values([scoringType.adp_column_name()], inplace = True)
    reg_df.drop_duplicates(subset = 'pfref_id', keep = 'first', inplace = True)
    reg_df['Flex'] = np.where(reg_df['FantPos'].isin(['RB','TE','WR']), 1, 0)
    reg_df['team'] = pd.Series([pd.NA] * len(reg_df), dtype=pd.StringDtype())
    reg_df['pick'] = np.nan
    return reg_df

def simulateLeagueAndDraft(year : int, 
                           temp : float, 
                           scoringType : ScoringType, 
                           numTeams: int = 10,
                           colSubset : List[str] = ['Player','Tm','Age','FantPos','Year','pfref_id','pred','var','var2','var_pred']) -> SnakeDraft:
    # st = time.time()
    colSubset = colSubset + [scoringType.adp_column_name(), scoringType.points_name()]
    league = League(numTeams, year)
    league.initTeamsDefault()

    playerPoolDf = initPlayerPoolDfFromRegDataset(year, scoringType, colSubset)
    playerPool = ADPPlayerPool(playerPoolDf, scoringType)
    snakeDraft = SnakeDraft(playerPool, league, year)
    draftOrder = snakeDraft.getDraftOrder(shuffle = False)
    
    # logger.info(f"Time to initialize league and pool: {time.time() - st}")

    # st = time.time()
    for num, team in enumerate(draftOrder):
        playerId : str = team.selectFromPool(snakeDraft.pool, num + 1, temp)
        team.finalizeSelection(playerId, snakeDraft.pool, num + 1)
    # logger.info(f"Time to draft: {time.time() - st}")
    return snakeDraft

if __name__ == '__main__':
    pd.options.display.max_columns = None
    reg_df = loadDatasetAfterBaseRegression(use_compressed = False)
    # print(reg_df)
    print(reg_df[reg_df['Year'] == 2023].head())
    col_subset = ['Player','Tm','Age','FantPos','Year','pfref_id','pred','var','var2','var_pred'] + [ScoringType.HPPR.adp_column_name(), ScoringType.HPPR.points_name()]
    a = initPlayerPoolDfFromRegDataset(2023, ScoringType.HPPR, col_subset, use_compressed = False)
    # print(a)
    snakeDraft = simulateLeagueAndDraft(2023, 4, ScoringType.HPPR)
    # for team in snakeDraft.league.teams:
        # print(team.roster)