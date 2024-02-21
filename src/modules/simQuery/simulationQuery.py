import pandas as pd
import pathlib
import os
import glob
import logging
import numpy as np
from io import BytesIO
from typing import List, Tuple
import boto3 
from functools import lru_cache
import requests
import json

from ...domain.common import ScoringType, loadDatasetAfterBaseRegression
from ...util.logger_config import setup_logger
LOG_LEVEL = logging.DEBUG
logger = setup_logger(__name__, level = LOG_LEVEL)


FIXED_YEAR = 2023
FIXED_SCORING_TYPE = ScoringType.HPPR
FIXED_COLS = ['Player','Tm','Age','FantPos','Year','pfref_id','pred','var','var2','var_pred'] + [FIXED_SCORING_TYPE.adp_column_name()]
IN_DEV = True
RUNNER_BASE_URL = 'http://127.0.0.1:5000'

# May need to revisit this, when we are no longer holding all data in memory
# def generateFirstPickSummaries():
#     for i in range(1, 11):
#         cli = SimulationQueryCLI(selectedTeam = i)
#         res = cli.runner.filterByTeamRound(i, 1, cli.runner.colSubset, [])
#         res.to_csv(f'../../data/regression/firstPickSummaries/team_{i}_round_1.csv', index = False)

def initPlayerPoolDfFromRegDataset(year: int, 
                                   scoringType : ScoringType, 
                                   colSubset : List[str]) -> pd.DataFrame:
    '''
    Let's store this locally with container
    '''
    logger.info("Initializing player pool from regression dataset...")
    reg_df = loadDatasetAfterBaseRegression()
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
        
def _readQueryableData(year: int = 2023):
    # Read anything in regression folder with queryable and year in name:
    pattern = f'../../data/regression/*queryableDraftPicks*{str(year)}*.csv'
    files = glob.glob(pattern)
    dfs = [pd.read_csv(file) for file in files]
    return pd.concat(dfs, ignore_index=True)

class ISimulationQueryRunner():
    def getPreselectInfo(self, teamNumber: int, roundNumber: int, conditions: List[str]) -> Tuple[int, pd.DataFrame, float]:
        raise NotImplementedError

    def makeSelection(self, idList: List[str], pickNum: int):
        raise NotImplementedError

class SimulationQueryService(ISimulationQueryRunner):
    def __init__(self, 
                 base_url = RUNNER_BASE_URL):
        self.base_url = base_url
    
    def getPreselectInfo(self, teamNumber: int, roundNumber: int, conditions: List[str]) -> Tuple[int, pd.DataFrame, float]:
        logger.info("Trying to get preselect info from service...")
        headers = {'Content-Type': 'application/json'}
        data = json.dumps({
            'teamNumber': teamNumber,
            'roundNumber': roundNumber,
            'conditions': conditions
        })

        logger.info("Sending POST request")
        response = requests.post(f'{self.base_url}/drafts/summary', headers=headers, data=data)
        if response.status_code == 200:
            # Assuming the response is JSON and contains the expected fields
            response_data = response.json()
            availablePlayers_json = response_data['availablePlayers']
            availablePlayers_df = pd.read_json(availablePlayers_json)
            return (response_data['pickNumber'], availablePlayers_df, response_data['expectedPoints'])
        else:
            # Handle error or empty response appropriately
            logger.error(f"Error or empty response: Status code {response.status_code}, Body: {response.text}")
            return None
    
    def makeSelection(self, idList: List[str], pickNum: int):
        logger.info("Trying to make selection from service...")
        headers = {'Content-Type': 'application/json'}
        data = json.dumps({
            'idList': idList,
            'pickNumber': pickNum
        })

        logger.info("Sending POST request to make selection...")
        response = requests.post(f'{self.base_url}/drafts/selection', headers=headers, data=data)
        if response.status_code == 200:
            # Assuming the response is JSON and contains the expected fields
            response_data = response.json()
            return response_data['selection']
        else:
            # Handle error or empty response appropriately
            logger.error(f"Error or empty response: Status code {response.status_code}, Body: {response.text}")
            return None

class SimulationQueryRunner(ISimulationQueryRunner):
    def __init__(self,
                 scoringType: ScoringType,
                 colSubset: List[str],
                 year: int):
        logger.info("Initializing SimulationQueryRunner...")
        self.scoringType = scoringType
        self.colSubset = colSubset
        self.year = year
        self.playerInfo = initPlayerPoolDfFromRegDataset(self.year, self.scoringType, self.colSubset)
        self.playerPicks = self._readQueryData()
        
    @lru_cache(maxsize=1)
    def _readQueryData(self, inDev: bool = IN_DEV) -> pd.DataFrame:
        logger.info("Reading queryable data...")
        if inDev:
            logger.info("In dev mode, so reading data from local...")
            return _readQueryableData(2023)
        logger.info("Not in dev mode, so reading data from S3...")
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket='ffwrapped')
        dfs = []
        if 'Contents' in response:
            for obj in response['Contents']:
                print("processing object in S3...")
                key = obj['Key']
                response = s3.get_object(Bucket='ffwrapped', Key=key)
                
                # Read the S3 object into a DataFrame
                df = pd.read_csv(BytesIO(response['Body'].read()))
                dfs.append(df)
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()
        
    def _getPickNumber(self, teamNumber: int, roundNumber: int, numTeams: int = 10):
        snakeNumber = (roundNumber + 1) // 2
        twoRoundTotal =  2 * numTeams # picks in 2 rounds
        snakeEnd = twoRoundTotal * snakeNumber
        snakeStart = snakeEnd - (twoRoundTotal - 1)
        if roundNumber % 2 == 1:  # Odd round
            return snakeStart + (teamNumber - 1)
        else:
            return snakeEnd - (teamNumber - 1)

    def _getPositionAverages(self, df: pd.DataFrame):
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

    def _limitLeagues(self, df: pd.DataFrame, conditions: List[Tuple[int, str]]):
        conditions = [i[1] for i in conditions]
        for i, condition in enumerate(conditions):
            col_name = f'cond{i+1}'
            # We use `eval` to evaluate the condition on the DataFrame
            df[col_name] = np.where(df.eval(condition), 1, 0)
        summ = df[['league'] + [f'cond{i+1}' for i in range(len(conditions))]].groupby('league').sum().reset_index()
        summ['sum'] = summ[[f'cond{i+1}' for i in range(len(conditions))]].sum(axis=1)
        filtered_leagues = summ.loc[summ['sum'] > len(conditions) - 1, 'league'].copy()
        return set(filtered_leagues)

    def getPreselectInfo(self, teamNumber: int, roundNumber: int, conditions: List[str]) -> Tuple[int, pd.DataFrame, float] :
        logger.info("Getting preselect info...")
        if 0 < roundNumber < 15 and 0 < teamNumber < 11:
            thisPickNumber = self._getPickNumber(teamNumber, roundNumber)
            logger.debug("Thispicknumber is set to: {}".format(thisPickNumber))
            colSubset1 = [i for i in self.colSubset[:] if i not in ['Age','var','var2']]
            availablePlayers = self.filterByTeamRound(teamNumber, roundNumber, colSubset1, conditions)
            expectedPoints = (availablePlayers['mean'] * availablePlayers['count']).sum() / availablePlayers['count'].sum()
        return thisPickNumber, availablePlayers, expectedPoints

    def filterByTeamRound(self, 
                          teamNumber: int, 
                          roundNumber: int, 
                          colSubset: List[str], 
                          conditions: List[str]):
        logger.debug("In filter by team round method of SQR...")
        # Limit all leagues to those that match initial conditions
        filteredLeagues = self._limitLeagues(self.playerPicks, conditions)
        playerPicks = self.playerPicks[self.playerPicks['league'].isin(filteredLeagues)].copy()
        
        # Return player picks for team and round, subject to conditions
        pick = self._getPickNumber(teamNumber, roundNumber)    
        playerPickSub = playerPicks[(playerPicks['team'] == f'team_{teamNumber}') & (playerPicks['pick'] == pick)].copy()
        playerPickSub = playerPickSub.merge(self.playerInfo[colSubset], on = ['pfref_id','FantPos'], how = 'left')
        
        # Summarize player picks by position
        positionAverages = self._getPositionAverages(playerPickSub)
        
        # Summarize player picks by player
        summaryDf = playerPickSub[colSubset + ['Total']].groupby(colSubset).agg({'Total' : ['count', 'mean']}).reset_index()
        summaryDf.columns = colSubset + ['count','mean']
        summaryDf['pct'] = (summaryDf['count'] / sum(summaryDf['count']) * 100).round(2)
        
        # # Concatenate
        summaryDf = pd.concat([summaryDf, positionAverages], ignore_index = True)
        summaryDf.sort_values('mean', ascending = False, inplace = True)    
        summaryDf[['mean','pct','pred','var_pred','AverageDraftPositionHPPR']] = summaryDf[['mean','pct','pred','var_pred','AverageDraftPositionHPPR']].round(2)
        return summaryDf[['Player','FantPos','Tm','pfref_id','count','mean','pct','pred','var_pred','AverageDraftPositionHPPR']]
        
    def makeSelection(self, idList: List[str], pickNum: int) -> str:
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


class SimulationQueryCLI():
    def __init__(self,
                 scoringType: ScoringType = FIXED_SCORING_TYPE,
                #  colSubset: List[str] = FIXED_COLS,
                 year: int = FIXED_YEAR,
                 selectedTeam: int = None):
        self.scoringType = scoringType
        # self.colSubset = colSubset + [self.scoringType.adp_column_name()]
        self.year = year
        # self.runner = SimulationQueryRunner(self.scoringType, self.colSubset, self.year)
        self.runner = SimulationQueryService()
        self.selectedTeam = selectedTeam
        # self.startup()

    def _selectTeam(self):
        logger.debug("In select team method of CLI...")
        selectedTeam = self.selectedTeam
        while not selectedTeam:
            selectedTeam = input("Which team (int value 1-10) would you like to draft for?\n")
            try:
                selectedTeam = int(selectedTeam)
            except:
                selectedTeam = None
            if not isinstance(selectedTeam, int) or selectedTeam < 1 or selectedTeam > 10:
                print("Invalid selection. Please enter a number between 1 and 10.")
                selectedTeam = None
        return selectedTeam

    def _doPreselectPrints(self, 
                            roundNumber: int,
                            pickNumber: int, 
                            availablePlayers: pd.DataFrame,
                            conditions: List[str],
                            expectedPoints: float):
        print(f"\nRound: {roundNumber} \ Pick: {pickNumber- (roundNumber - 1) * 10} \ Overall: {pickNumber}")
        print(f"Team {self.selectedTeam}, please make your selection for Round {roundNumber}")
        print(f"Previous selections {conditions}")
        print(f"Expected max points for your team going into this selection: {expectedPoints}")
        print(f"Available samples backing this draft path: {availablePlayers['count'].sum() / 2}")
        print(availablePlayers)

    def requestSelection(self, roundNumber: int, availablePlayers: pd.DataFrame, conditions: List[str]):
        # Query user input until valid seleciton is made
        enteredPlayerId = None
        enteredList = []
        goBack = False
        while not enteredPlayerId:
            enteredPlayerId = input("Who do you choose? Please enter pfref_id of your selection...\n")
            if enteredPlayerId == 'goback':
                if roundNumber == 1:
                    print("Cannot go back further. Please enter a valid selection...")
                    enteredPlayerId = None
                    continue
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
        return roundNumber, enteredList, goBack
            
    def startup(self):
        logger.debug("In startup method of CLI...")
        print("Welcome to this sample draft simulation! We will be simulating a 10-team HPPR draft for the year2023.")
        self.selectedTeam = self._selectTeam()
        roundNumber = 1
        conditions = []
        while roundNumber < 15:
            logger.info("Processing round number: {}".format(roundNumber))
            thisPickNumber, availablePlayers, expectedPoints = self.runner.getPreselectInfo(self.selectedTeam, roundNumber, conditions)
            self._doPreselectPrints(roundNumber, thisPickNumber, availablePlayers, conditions, expectedPoints)
            
            roundNumber, enteredList, goBack = self.requestSelection(roundNumber, availablePlayers, conditions)
            if goBack:
                print("I am going back now!")
                continue
            # Make selection
            selection = self.runner.makeSelection(enteredList, thisPickNumber)
            conditions.append((thisPickNumber, selection))
            logger.debug(f"Conditions: {conditions}")
            roundNumber += 1

   
if __name__ == '__main__':
    path = pathlib.Path(__file__).parent.resolve()
    os.chdir(path)

    # generateFirstPickSummaries()
    cli = SimulationQueryCLI()
    cli.startup()

    # cli.test()
    # runner = SimulationQueryRunner()
    # a = runner._readQueryData(IN_DEV)
    # print(a)