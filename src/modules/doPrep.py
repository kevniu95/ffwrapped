import os
import pathlib
from typing import List, Dict, Callable
from abc import ABC, abstractmethod
import logging

import numpy as np
import pandas as pd
import pickle

from ..domain.common import ScoringType, thisFootballYear
from ..util.logger_config import setup_logger

LOG_LEVEL = logging.INFO
logger = setup_logger(__name__, level = LOG_LEVEL)

PFREF_COL_NAMES = ['Rk', 'Player', 'Tm', 'FantPos', 'Age', 'G', 'GS', 'PassCmp', 'PassAtt', 'PassYds',
                    'PassTD', 'PassInt', 'RushAtt', 'RushYds', 'RushY/A', 'RushTD', 'RecTgt', 'Rec', 'RecYds', 'RecY/R',
                    'RecTD', 'Fmb', 'FL', 'TD', '2PM', '2PP', 'FantPt', 'PPR', 'DKPt', 'FDPt',
                    'VBD', 'PosRank', 'OvRank', 'Year', 'pfref_id']

class PointsConverter():
    def __init__(self, scoringType : ScoringType = ScoringType.NPPR):
        self.scoringType = scoringType
        self.score_dict = {'PassYds' : 0.04,
                            'PassTD' : 4,
                            'PassInt' : -2,
                            'RushYds' : 0.1,
                            'RushTD' : 6,
                            'Rec': self.scoringType.value,
                            'RecYds' : 0.1,
                            'RecTD' : 6,
                            'FL' : -2,
                            '2PM' : 2,
                            '2PP' : 2}

    def _score_row(self, row : pd.Series) -> float:
        sum = 0.0
        for cat, score in self.score_dict.items():
            addval = float(row[cat]) * score
            sum += addval
        return sum

    def calculate_points(self, 
                    fpts_dict : Dict[str, pd.DataFrame], 
                    pfref_colNames : Dict[str, str] = PFREF_COL_NAMES) -> pd.DataFrame:
        # 1. Concatenate, rename cols, drop filler rows, reset index
        df = pd.concat(fpts_dict.values())
        df.columns = pfref_colNames
        df = df.drop(df[df['Player'] == 'Player'].index) 
        df = df.reset_index().drop(['index','Rk'], axis = 1)    
        
        # 2. Convert numerics, fill nas with 0, then score
        score_cols = list(self.score_dict.keys()) + ['FantPt', 'PPR']
        df[score_cols] = df[score_cols].apply(pd.to_numeric)
        score_dict2 = {k : 0 for (k, v) in self.score_dict.items()}
        df.fillna(score_dict2, inplace=True)
        
        # 3. Score
        col_name = self.scoringType.points_name()
        df[col_name] = df.apply(self._score_row, axis = 1)
        if col_name == 'Pts_PPR':
            assert len(df[(df['Pts_PPR'] - df['PPR']) > 0.1]) == 0

        # 4. Clean player names of '*' and '+'
        df['Player'] = df['Player'].str.replace(r'[\*\+]', '', regex=True).str.strip()

        # 5. Limit to guys with positions, everyone without position: these guys always have 0 or less pts scored
        df = df[df['FantPos'].notnull()].copy()
        return df

class PreparationStep():
    def __init__(self, name : str, method : Callable):
        self.name = name
        self.method = method
    
    def execute(self, *args, **kwargs) -> pd.DataFrame:
        return self.method(*args, **kwargs)

class Dataset(ABC):
    def __init__(self, sources : List[str], prepSteps : List[PreparationStep] = None):
        '''
        A dataset has following propertiess
            - sources : List of file paths containing soruce data
            - prepSteps : Ordered list of PreparationSteps, which transform a pandas DataFrame
                - to another pandas DataFrame
                - Steps can be excluded and potentially re-ordered, depending on specific implementation
            - prepStepMapping : Dictionary mapping name of steps in prepSteps to underlying method
        '''
        self.sources = sources
        self.prepSteps = self.setDefaultPrepSteps(prepSteps)
        
    def setDefaultPrepSteps(self, prepSteps : List[PreparationStep]) -> List[PreparationStep]:
        if prepSteps:
            return prepSteps
        return []
    
    @abstractmethod
    def loadData() -> pd.DataFrame:
        pass

    def performSteps(self) -> pd.DataFrame:
        logger.info(f"Performing preparation steps on {self.__class__.__name__}")
        df = self.loadData()
        for step in self.prepSteps:
            logger.info(f"Performing step: {step.name}")
            df = step.execute(df)
        return df

class PointsDataset(Dataset):
    def __init__(self, 
                 sources : List[str], 
                 scoringType : ScoringType,
                 pointsConverter : PointsConverter = None, 
                 prepSteps : List[PreparationStep] = None,
                 currentYear : int = thisFootballYear(),
                 currentRosterDf : pd.DataFrame = None):
        self.currentRosterDf = currentRosterDf
        self.currentYear = currentYear
        self.scoringType  = scoringType
        self.pointsConverter = pointsConverter
        if not ((self.pointsConverter) and (self.pointsConverter.scoringType == scoringType)):
            print("PointsConverter was not defined or type doesn't match specified ScoringType in this constructor."
                    + "Creating new one...")
            self.pointsConverter = PointsConverter(scoringType)
        super().__init__(sources, prepSteps)        

    def setDefaultPrepSteps(self, prepSteps : List[PreparationStep]) -> List[PreparationStep]:
        if prepSteps:
            super().setDefaultPrepSteps(prepSteps)
        prepSteps = [PreparationStep('Group Multi-Team Players', self._groupMultiTeamPlayers),
                     PreparationStep('Filter out predicted year from data', self._filterPredictYear),
                     PreparationStep('Get Previous Year Data', self._createPreviousYear)]
        if self.currentRosterDf is not None:
            prepSteps.append(PreparationStep(f"Append {self.currentYear} roster to dataset", self._addCurrentRosters))
        prepSteps.extend([PreparationStep('Create QB Change Info', self._create_qb_chg),
                              PreparationStep('Get point share at position', self._getPtShare)])
        return prepSteps

    def loadData(self) -> pd.DataFrame:
        dfDict = {}
        for source in self.sources:
            with open(source, 'rb') as handle:
                dfDict.update(pickle.load(handle))
        return self.pointsConverter.calculate_points(dfDict)

    def _groupMultiTeamPlayers(self, df : pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(['pfref_id', 'Year', 'PPR'], ascending=[True, True, False])

        def _aggregate_players(df):
            # Create a flag for groups with more than one member
            df['flag'] = df.groupby(['pfref_id', 'Year'])['Tm'].transform('size') > 1
            # Modify 'Tm' based on the flag
            df.loc[df['flag'], 'Tm'] = df.loc[df['flag'], 'Tm'] + 'TM'
            # Drop duplicates based on 'pfref_id' and 'Year' to mimic selecting the first row
            result = df.drop_duplicates(subset=['pfref_id', 'Year']).drop(columns=['flag'])
            return result
        return _aggregate_players(df)
    
    def _filterPredictYear(self, df: pd.DataFrame) -> pd.DataFrame():
        return df[df['Year'] != self.currentYear]
        
    def _addCurrentRosters(self, df : pd.DataFrame) -> None:
        """
        Adds information about rookies from current year rosters
        Outer merge may add new players
        
        Fill in following if new player 
            1. Player 
            2. Team
            3. Fantasy Position

        Also add rookie/draft info where applicable
        """

        cols = list(df.columns) + ['rookie', 'draftPick', 'Yrs']

        # Merge by pfref_id, fill in other info after
        merged = df.merge(self.currentRosterDf, on = ['pfref_id','Year'], how = 'outer')
        # print(merged[merged['Player_x'] == 'John Kuhn'])
        for i in ['Player', 'Tm', 'FantPos']:
            merged[i] = merged[i + '_x'].fillna(merged[i + '_y'])
        final_df = merged[cols].copy()
        # Re-do changedTeam var for 2023 guys
        final_df['changedTeam'] = np.where((final_df['Tm'] == merged['PrvTm']) | (merged['PrvTm'].isnull()), 0, 1)
        final_df = final_df[final_df['Player'].notnull()]
        return final_df.drop_duplicates()
        
    def _createPreviousYear(self, pts_df_base : pd.DataFrame) -> pd.DataFrame:
        """
        Initial fields:
            1. Player
            2. Team
            3. Age
            4. Fantasy Position
            5. Year
            6. pfref_id
            7. Points
        
        Added fields:
            1. Previous year team
            2. Previous year age
            3. Previous year
            4. Previous year points
            5. Missing last year flag
            6. Team change flag
            7. Year born
        """
        scoring_var = self.scoringType.points_name()
        # 1. Create template
        predTemplate = pts_df_base[['Player', 'Tm', 'Age', 'FantPos', 'Year', 'pfref_id', scoring_var]]
        # print(len(pts_df_base[pts_df_base['Year'] == 2022]))
        
        if LOG_LEVEL == logging.DEBUG:
            print(f"1. Shape of og dataset is: {predTemplate.shape}\n") 
        
        # 2. Merge on last year's results
        prvYr = predTemplate[['Tm', 'Age', 'Year', 'pfref_id', scoring_var]].copy()
        prvYr.rename(columns = {'Year' : 'PrvYear', 
                                'Tm' : 'PrvTm', 
                                scoring_var : 'Prv' + scoring_var, 
                                'Age' : 'PrvAge'}, 
                                inplace = True)
        prvYr['Year'] = prvYr['PrvYear'] + 1
        prvYr['PrvAge'] = pd.to_numeric(prvYr['PrvAge'])
        merged = predTemplate.merge(prvYr, on = ['pfref_id', 'Year'], how = 'outer', indicator= 'foundLastYearStats')
        
        if LOG_LEVEL == logging.DEBUG:
            print(predTemplate.shape)
            print(prvYr.shape)
            print("2. This is the shape after doing outer join with previous years' data")
            print(merged.shape)
            print(merged['foundLastYearStats'].value_counts())
            print()
        
        # 3. Remove right_only obs that aren't from 2022
            print(f"3. In total, {len(merged[merged['foundLastYearStats'] == 'right_only'])} observations are players with some data, "
                + "but no previous year statistics")
            print(f"-Of these, {len(merged[merged['PrvYear'] == (self.currentYear - 1)])} observations are associated" 
                + f" with year {self.currentYear}")
            print("\t-These observations don't have a 'y-value' for regression, but this is the year we are trying to predict, so OK to exclude")
            print(f"-Remove remaining {len(merged[(merged['PrvYear'] != (self.currentYear - 1)) & (merged['foundLastYearStats'] == 'right_only')])} observations")
            print("\t-These observations don't have a 'y-value' for regression, only 'x-values', so ok to delete")
            print("The easiest way to look at this is as follows")
            print("\t-All players-year observations prior to this predicted year are appended with previous year's if it's there")
            print("\t-If previous year is not there, the player is still included")
            print("\t-In addition to these players, we also add players for this predicted year, who will ONLY have previous year's data")
            print()

        merged = merged[(merged['Year'] == self.currentYear) | (merged['foundLastYearStats'] != 'right_only')]
        
        # 4. Create found last year flag
        # This will help distinguish rookies and other players not in data
        merged['PrvYear'] = merged['Year'] - 1

        # 5. Examine composition of remaining observations  
        # Left_only and both are needed for regression - excludes 2013 observations
        # Right_only needed for prediction - excludes non-2022 observations (right-only's in OG data)
        if LOG_LEVEL == logging.DEBUG:
            print(f"# players with both years' data:\n-{len(merged[merged['foundLastYearStats'] == 'both'])}")
            print(f"# players with this year, not last year's data:\n-{len(merged[merged['foundLastYearStats'] == 'left_only'])}")
            print(f"# players with last year, not this year's data (i.e., we predict this year):\n-{len(merged[merged['foundLastYearStats'] == 'right_only'])}")
            print(merged['foundLastYearStats'].value_counts())
            print()
        merged['missingLastYear'] = np.where(merged['foundLastYearStats']=='left_only', 1, 0)
        merged['changedTeam'] = np.where((merged['Tm'] == merged['PrvTm']) | (merged['PrvTm'].isnull()), 0, 1)
        merged['Age'] = merged['PrvAge'] + 1

        age_df = merged[['Player', 'pfref_id', 'Year','Age']].drop_duplicates()
        age_df['year_born'] = age_df['Year'].astype(int) - age_df['Age']
        age_df = age_df[age_df['year_born'].notnull()]
        assert len(age_df.drop_duplicates('pfref_id')) == len(age_df[['pfref_id','year_born']].drop_duplicates())
        age_df = age_df[['pfref_id','year_born']].drop_duplicates()
        merged = merged.merge(age_df, on ='pfref_id', how = 'left')
        merged['Age'] = merged['Year'] - merged['year_born']
        if LOG_LEVEL == logging.DEBUG:
            print(merged.head())
        
        return merged.drop('foundLastYearStats', axis = 1)
    
    def _create_qb_chg(self, df : pd.DataFrame) -> pd.DataFrame:
        """
        Captures qb change info, like:
            1. old QB points
            2. new QB points
            3. difference in points
            4. flag for new QB
        """
        scoring_var : str = self.scoringType.points_name()
        # Do this a new way: separately merge on last year's QB and then this year's QB
        qb_then = df.loc[df['FantPos'] == 'QB', ['Tm','Year', scoring_var]]
        qb_then = qb_then.groupby(['Tm', 'Year'], as_index = False).max()[['Tm', 'Year', scoring_var]]
        qb_then.rename(columns = {scoring_var : scoring_var + '_old_qb'}, inplace= True)
        qb_then = qb_then[qb_then['Tm'].str[-2:] != 'TM']
        
        new_df = df.merge(qb_then[['Tm', 'Year', scoring_var + '_old_qb']], 
                        left_on = ['PrvTm', 'PrvYear'], 
                        right_on = ['Tm','Year'], 
                        how = 'left', 
                        suffixes = [None, '_y'])
        new_df.drop('Tm_y', axis = 1, inplace = True)
        new_df.drop('Year_y', axis = 1, inplace = True)
        
        prv_scoring_var = 'Prv' + scoring_var
        qb_now = df.loc[df['FantPos'] == 'QB', ['Tm', 'Year', prv_scoring_var]]
        qb_now = qb_now.groupby(['Tm', 'Year'], as_index = False).max()[['Tm','Year', prv_scoring_var]]
        qb_now = qb_now.rename(columns = {prv_scoring_var : scoring_var + '_new_qb'})
        qb_now = qb_now[qb_now['Tm'].str[-2:] != 'TM']
        new_df = new_df.merge(qb_now[['Tm','Year', scoring_var + '_new_qb']], 
                            left_on = ['Tm','Year'], 
                            right_on = ['Tm','Year'], 
                            how = 'left',
                            suffixes = [None, '_y'])
        
        new_df['qb_diff'] = new_df[scoring_var + "_new_qb"] - new_df[scoring_var + "_old_qb"]

        # Pick up where there is no new qb going into season
        new_df['noNewQb'] = np.where((new_df['Pts_HPPR_new_qb'].isnull()) & (new_df['Tm'].str[-2:] != 'TM'), 1 ,0)
        return new_df
        
    def _getPtShare(self, df : pd.DataFrame) -> pd.DataFrame:
        """
        Calculate:
            1. Previous year team points at position
                i. Last year's points, aggregated over position players on this year's team
            2. Number of players at position this year
            3. Share of previous year points at position
        """
        scoring_var : str = self.scoringType.points_name()
        prv_scoring_var = 'Prv' + scoring_var
        df['ones'] = 1
        df_tm = df.groupby(['Tm','FantPos','Year'], as_index = False).sum()[['Tm','FantPos','Year', prv_scoring_var, 'ones']]
        df_tm = df_tm.rename(columns = {prv_scoring_var : 'PrvYrTmPtsAtPosition', 'ones' : 'PlayersAtPosition'})
        df_tm = df_tm[df_tm['Tm'].str[-2:] != 'TM']
        
        df = df.merge(df_tm, on = ['Tm','FantPos','Year'], how = 'left')
        df['PrvYrPtsShare'] = df[prv_scoring_var] / df['PrvYrTmPtsAtPosition'] 
        df.drop('ones', axis = 1, inplace = True)
        return df

class ADPDataset(Dataset):
    def __init__(self, 
                 scoringType : ScoringType,
                 sources : List[str] = None, 
                 prepSteps : List[PreparationStep] = None):
        super().__init__(sources, prepSteps)
        self.scoringType = scoringType

    def setDefaultPrepSteps(self, prepSteps : List[PreparationStep]) -> List[PreparationStep]:
        if prepSteps:
            super().setDefaultPrepSteps(prepSteps)
        prepSteps = [PreparationStep('Drop 2014 and 2015 from ADP', self._dropSmallerDatasets)]
        return prepSteps
    
    def _dropSmallerDatasets(self, df : pd.DataFrame) -> pd.DataFrame:
        return df[~df['Year'].isin([2014, 2015])]

    def _loadPickleDict(self) -> Dict[str, pd.DataFrame]:
        pickleDict = {}
        for source in self.sources:
            if source.endswith('adp.p') or source.endswith('adp_full.p'):
                pickleDict[ScoringType.PPR] = pd.read_pickle(source)
            elif source.endswith('adp_nppr.p') or source.endswith('adp_nppr_full.p'):
                pickleDict[ScoringType.NPPR] = pd.read_pickle(source)
            else:
                raise Exception("Only pickles with suffix 'adp.p' or 'adp_nppr.p' are acceptable right now!")
        return pickleDict
        
    def loadData(self) -> pd.DataFrame:
        pickleDict = self._loadPickleDict()
        ppr_set = pickleDict[ScoringType.PPR]
        nppr_set = pickleDict[ScoringType.NPPR]
        if self.scoringType == ScoringType.PPR:
            return ppr_set
        elif self.scoringType == ScoringType.NPPR:
            return nppr_set
        elif self.scoringType == ScoringType.HPPR:
            adp_merged = ppr_set.merge(nppr_set, on = ['Name','Year','Team','Position'], how = 'outer')
            
            adp_merged['AverageDraftPositionHPPR'] = (adp_merged['AverageDraftPositionPPR'].fillna(adp_merged['AverageDraftPosition']) 
                                                      + adp_merged['AverageDraftPosition'].fillna(adp_merged['AverageDraftPositionPPR'])) / 2
            keep_cols = ['Name','Year','Team','Position','AverageDraftPositionHPPR']
            adp_merged = adp_merged[keep_cols].groupby(['Name','Year','Team','Position'], as_index = False).mean()
            return adp_merged[keep_cols]

class RosterDataset(Dataset):
    def __init__(self,
                 sources : List[str],
                 prepSteps: List[PreparationStep] = None,
                 currentYear : int = thisFootballYear()):
        super().__init__(sources, prepSteps)
        self.currentYear = currentYear
    
    def setDefaultPrepSteps(self, prepSteps : List[PreparationStep]) -> List[PreparationStep]:
        if prepSteps:
            super().setDefaultPrepSteps(prepSteps)
        prepSteps = [PreparationStep('Fix team name abbreviations', self._getAbbreviation),
                     PreparationStep('Extract roster info', self._extractRosterInfo)]
        return prepSteps
    
    def loadData(self) -> pd.DataFrame:
        dfs = []
        for source in self.sources:
            dfs.append(pd.read_pickle(source))
        df = pd.concat(dfs)
        df.rename(columns = {'tm' : "Tm", 'ID' : 'pfref_id', 'Pos' : 'FantPos'}, inplace = True)
        df['Player'] = df['Player'].str.replace(r'\(.*?\)', '', regex=True)
        return df.loc[df['FantPos'].isin(['RB','TE','WR','QB']), ['Player', 'Tm', 'FantPos', 'Year', 'pfref_id', 'Yrs','Drafted (tm/rnd/yr)']].copy()
    
    def _getAbbreviation(self, df : pd.DataFrame) -> pd.DataFrame:
        """
        Calculate:
            1. Team position abbreviation
        """
        abbr_df = pd.read_csv('../../data/imports/helpers/abbreviations.csv')
        mapping = dict(zip(abbr_df['pfr'].str.lower(), abbr_df['pfr_schedule']))
        df.replace({'Tm' : mapping}, inplace = True)
        return df
    
    def _extractRosterInfo(self, df : pd.DataFrame) -> pd.DataFrame:
        """
        Calculate:
            1. Rookie flag
            2. Draft pick
            3. Extract years played
        """
        df['rookie'] = np.where(df['Yrs'] == 'Rook', 1, 0)
        df['draftPick'] = df['Drafted (tm/rnd/yr)'].str.split('/').str[2].str.extract(r'(\d+)')
        df['Yrs'] = df['Yrs'].replace('Rook', 0).astype(int)
        return df[['Player', 'Tm', 'FantPos', 'Year', 'pfref_id', 'rookie', 'draftPick','Yrs']].drop_duplicates()

if __name__ == '__main__':
    pd.options.display.max_columns = None
    path = pathlib.Path(__file__).parent.resolve()
    os.chdir(path)

    SCORING = ScoringType.HPPR

    # ======
    # Roster
    # ======
    roster_source = '../../data/imports/created/rosters.p'
    rd = RosterDataset([roster_source])
    # rd_performed = rd.performSteps()
    # print(a[a['Year'] == 2023])

    # =======
    # Points
    # =======
    pc = PointsConverter(SCORING)
    points_sources = ['../../data/imports/created/points.p']
    a = pd.read_pickle(points_sources[0])
    pointsDataset = PointsDataset(points_sources, SCORING, pc, currentRosterDf= rd_performed)
    df = pointsDataset.loadData()
    # res = pointsDataset._groupMultiTeamPlayers(df)
    # res = pointsDataset._filterPredictYear(res, 2023)
    # res = pointsDataset._createPreviousYear(res)
    # print(df.shape)
    # print(res.shape)
    # print(res.duplicated(['pfref_id','Year'], keep=False).sum())
    
    # print(pt[pt['Player'].str.contains('Beau')])
    # print(pt)
    
    # =======
    # ADP
    # =======
    adp_sources = ['../../data/imports/created/adp_full.p',
                   '../../data/imports/created/adp_nppr_full.p']
    ad = ADPDataset(SCORING, adp_sources)
    # adres = ad.performSteps()
    # print(adres)

    # a = pd.read_pickle('../../data/imports/created/rosters.p')
    # print(a)