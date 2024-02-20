import pandas as pd
import pathlib
import os
import glob
import logging
import numpy as np
from typing import List, Tuple
from ..domain.common import loadDatasetAfterBaseRegression, ScoringType
from .draft import initPlayerPoolDfFromRegDataset
# from .stageData import *
from ..util.logger_config import setup_logger
LOG_LEVEL = logging.INFO
logger = setup_logger(__name__, level = logging.INFO)

    
def limitLeagues(df: pd.DataFrame, conditions: List[Tuple[int, str]]):
    conditions = [i[1] for i in conditions]
    for i, condition in enumerate(conditions):
        col_name = f'cond{i+1}'
        # We use `eval` to evaluate the condition on the DataFrame
        df[col_name] = np.where(df.eval(condition), 1, 0)
    summ = df[['league'] + [f'cond{i+1}' for i in range(len(conditions))]].groupby('league').sum().reset_index()
    summ['sum'] = summ[[f'cond{i+1}' for i in range(len(conditions))]].sum(axis=1)
    filtered_leagues = summ.loc[summ['sum'] > len(conditions) - 1, 'league'].copy()
    return set(filtered_leagues)

def getPositionAverages(df: pd.DataFrame):
    keep_vars = ['FantPos','Total','pred','var_pred','AverageDraftPositionHPPR']
    newDf = df[keep_vars].groupby('FantPos').agg({'Total' : ['count','mean'],
                                                            'pred' : 'mean',
                                                            'var_pred' : 'mean',
                                                            'AverageDraftPositionHPPR' : 'mean'}).reset_index()
    newDf.columns = ['FantPos','count','mean', 'pred', 'var_pred','AverageDraftPositionHPPR']
    newDf['Player'] = newDf['FantPos'] + ' - Average' 
    newDf['pct'] = (newDf['count'] / sum(newDf['count']) * 100).round(2)
    newDf['pfref_id'] = 'Avg' + newDf['FantPos']
    return newDf
    
def filterByTeamRound(teamNumber: int, 
                      roundNumber: int, 
                      playerPicks: pd.DataFrame, 
                      playerInfo: pd.DataFrame,
                      colSubset : List[str],
                      conditions : List[str]) -> pd.DataFrame:
    # Limit all leagues to those that match initial conditions
    filteredLeagues = limitLeagues(playerPicks, conditions)
    playerPicks = playerPicks[playerPicks['league'].isin(filteredLeagues)].copy()
    
    # Return player picks for team and round, subject to conditions
    pick = pickNumber(teamNumber, roundNumber)    
    playerPickSub = playerPicks[(playerPicks['team'] == f'team_{teamNumber}') & (playerPicks['pick'] == pick)].copy()
    playerPickSub = playerPickSub.merge(playerInfo[colSubset], on = ['pfref_id','FantPos'], how = 'left')
    
    # Summarize player picks by position
    positionAverages = getPositionAverages(playerPickSub)
    
    # Summarize player picks by player
    summaryDf = playerPickSub[colSubset + ['Total']].groupby(colSubset).agg({'Total' : ['count', 'mean']}).reset_index()
    summaryDf.columns = colSubset + ['count','mean']
    summaryDf['pct'] = (summaryDf['count'] / sum(summaryDf['count']) * 100).round(2)
    
    # # Concatenate
    summaryDf = pd.concat([summaryDf, positionAverages], ignore_index = True)
    summaryDf.sort_values('mean', ascending = False, inplace = True)    
    summaryDf[['mean','pct','pred','var_pred','AverageDraftPositionHPPR']] = summaryDf[['mean','pct','pred','var_pred','AverageDraftPositionHPPR']].round(2)
    return summaryDf[['Player','FantPos','Tm','pfref_id','count','mean','pct','pred','var_pred','AverageDraftPositionHPPR']]
    
def pickNumber(teamNumber, roundNumber, numTeams = 10):
    snakeNumber = (roundNumber + 1) // 2
    twoRoundTotal =  2 * numTeams # picks in 2 rounds
    snakeEnd = twoRoundTotal * snakeNumber
    snakeStart = snakeEnd - (twoRoundTotal - 1)
    if roundNumber % 2 == 1:  # Odd round
        return snakeStart + (teamNumber - 1)
    else:
        return snakeEnd - (teamNumber - 1)

def makeSelection(idList: List[str], pickNum: int):
    if len(idList) == 1 and idList[0][:3] == 'Avg' and idList[0][-2:] in ['QB','RB','WR','TE']:
        print("Selected a position group {} - Average".format(idList[0][-2:]))
        return f'''(pick == {pickNum}) & (FantPos == '{idList[0][-2:]}')'''
    
    secondCondition = "("
    for id in idList:
        if id != idList[0]:
            secondCondition += " | "
        secondCondition += f'''pfref_id == "{id}"'''
    secondCondition += ")"
    return f'''(pick == {pickNum}) & {secondCondition}'''


def _readQueryableData(year: int = 2023):
    # Read anything in regression folder with queryable and year in name:
    pattern = f'../../data/regression/*queryableDraftPicks*{str(year)}*.csv'
    files = glob.glob(pattern)
    dfs = [pd.read_csv(file) for file in files]
    return pd.concat(dfs, ignore_index=True)


class SimulationQueryRunner():
    def __init__(self):
        self.scoringType = FIXED_SCORING_TYPE
        self.colSubset = FIXED_COLS
        self.colSubset = self.colSubset + [self.scoringType.adp_column_name()]
        self.year = FIXED_YEAR
        self.playerInfo = initPlayerPoolDfFromRegDataset(self.year, self.scoringType, self.colSubset)

    def _readQueryData(self):
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket='ffwrapped')
        dfs = []
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                response = s3.get_object(Bucket='ffwrapped', Key=key)
                
                # Read the S3 object into a DataFrame
                df = pd.read_csv(BytesIO(response['Body'].read()))
                dfs.append(df)
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()




def makeSelections(teamNumber: int, year: int = 2023):
    scoringType = ScoringType.HPPR
    colSubset = ['Player','Tm','Age','FantPos','Year','pfref_id','pred','var','var2','var_pred']
    colSubset = colSubset + [scoringType.adp_column_name()]

    # Read in player data
    playerInfo = initPlayerPoolDfFromRegDataset(2023, scoringType, colSubset)
    playerPicks = _readQueryableData(2023)

    conditions = []
    roundNumber = 1
    while roundNumber < 15:
        thisPick = pickNumber(teamNumber, roundNumber)
        print(f"\nRound: {roundNumber} \ Pick: {thisPick- (roundNumber - 1) * 10} \ Overall: {thisPick}")
        print(f"Team {teamNumber}, please make your selection for Round {roundNumber}")
        colSubset1 = ['Player','Tm','FantPos','Year','pfref_id','pred','var_pred', scoringType.adp_column_name()]
        availablePlayers = filterByTeamRound(teamNumber, roundNumber, playerPicks, playerInfo, colSubset1, conditions)
        expectedPoints = (availablePlayers['mean'] * availablePlayers['count']).sum() / availablePlayers['count'].sum()
        print(f"Previous selections {conditions}")
        print(f"Expected max points for your team going into this selection: {expectedPoints}")
        print(f"Available samples backing this draft path: {availablePlayers['count'].sum() / 2}")
        print(availablePlayers)
        
        # Query user input until valid seleciton is made
        enteredPlayerId = None
        goBack = False
        while not enteredPlayerId:
            enteredPlayerId = input("Who do you choose? Please enter pfref_id of your selection...\n")
            if enteredPlayerId == 'goback':
                roundNumber -= 1
                conditions.pop()
                print("Reversing last draft pick...")
                goBack = True 
                break
            enteredList = enteredPlayerId.split(' ')
            for entered in enteredList:
                if entered not in set(availablePlayers['pfref_id']):
                    print(f"Invalid selection: {entered}. Please enter pfref_id of your selection and try again...\n")
                    enteredPlayerId = None
                    break
        
        if goBack:
            print("I am going back now!")
            continue
        # Make selection
        selection = makeSelection(enteredList, thisPick)
        conditions.append((thisPick, selection))
        roundNumber += 1
        
   
if __name__ == '__main__':
    path = pathlib.Path(__file__).parent.resolve()
    os.chdir(path)

    print("Welcome to this sample draft simulation! We will be simulating a 10-team HPPR draft for 2023.")
    selectedTeam = None
    while not selectedTeam:
        selectedTeam = input("Which team (int value 1-10) would you like to draft for?\n")
        try:
            selectedTeam = int(selectedTeam)
        except:
            selectedTeam = None
        if not isinstance(selectedTeam, int) or selectedTeam < 1 or selectedTeam > 10:
            print("Invalid selection. Please enter a number between 1 and 10.")
            selectedTeam = None

    makeSelections(selectedTeam, 2023)

# Test pickNumber
# last = 0
# for rd in range(1, 16):
#     for tm in range(1, 11):
#         if rd % 2 == 0:
#             tm = 11 - tm
#         val = pick_number(tm, rd)
#         print(val)
#         assert val == last + 1
#         last = val 
        

# Example usage:
# team = 1
# round = 2
# print("Pick number:", pick_number(team, round))  # Output: 1