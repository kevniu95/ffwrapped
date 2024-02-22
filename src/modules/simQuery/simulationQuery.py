import glob
import logging
import numpy as np
from io import BytesIO, StringIO
from typing import List, Tuple, Set
import pandas as pd
import boto3 
from functools import lru_cache
import requests
import json

from ...domain.common import ScoringType, loadDatasetAfterBaseRegression
from ...util.logger_config import setup_logger
LOG_LEVEL = logging.INFO
logger = setup_logger(__name__, level = LOG_LEVEL)

IN_DEV = True
RUNNER_BASE_URL = 'https://8fktyt5czr.us-east-2.awsapprunner.com'
# Simply-deployed service using AWS App Runner - will change this to ECS deployment with permanent URL

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
        logger.debug("Trying to get preselect info from service...")
        headers = {'Content-Type': 'application/json'}
        data = json.dumps({
            'teamNumber': teamNumber,
            'roundNumber': roundNumber,
            'conditions': conditions
        })

        logger.debug(f"Sending POST request with this data {conditions}")
        response = requests.post(f'{self.base_url}/drafts/summary', headers=headers, data=data)
        if response.status_code == 200:
            # Assuming the response is JSON and contains the expected fields
            response_data = response.json()
            availablePlayers_json = response_data['availablePlayers']
            availablePlayers_df = pd.read_json(StringIO(availablePlayers_json))
            return (response_data['pickNumber'], availablePlayers_df, response_data['expectedPoints'])
        else:
            # Handle error or empty response appropriately
            logger.error(f"Error or empty response: Status code {response.status_code}, Body: {response.text}")
            return None
    
    def makeSelection(self, idList: List[str], pickNum: int):
        logger.debug("Trying to make selection from service...")
        headers = {'Content-Type': 'application/json'}
        data = json.dumps({
            'idList': idList,
            'pickNumber': pickNum
        })

        logger.debug("Sending POST request to make selection...")
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

    def _applySingleLimitation(self, df: pd.DataFrame, condition: Tuple[int, str]) -> Set[str]:
        strCondition = condition[1]
        filteredLeagues = df.loc[df.eval(strCondition),'league'].unique()
        return set(filteredLeagues)

    @lru_cache(maxsize=64)
    def _limitLeagues(self, conditions: Tuple[Tuple[int, str]]) -> Set[str]:
        df = self.playerPicks
        if len(conditions) == 0:
            return set(df['league'].unique())
        elif len(conditions) == 1:
            return self._applySingleLimitation(df, conditions[-1])
        else:
            filteredLeagues = self._limitLeagues(conditions[:-1])
            currDf = df[df['league'].isin(filteredLeagues)].copy()
            return self._applySingleLimitation(currDf, conditions[-1])

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
        # logger.info(f"Here are conditions before tuple-izing: {conditions}")
        conditionsTuple = tuple(conditions)
        # logger.info(f"here are conditions as a tuple {conditionsTuple}")
        filteredLeagues = self._limitLeagues(conditionsTuple)
        # logger.info(f"Here are filtered leagues: {filteredLeagues}")
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