import os
import pathlib
from typing import List, Dict, Callable
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import pickle


from ..domain.common import ScoringType, thisFootballYear

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

    def _score_row(self, row):
        sum = 0.0
        for cat, score in self.score_dict.items():
            addval = float(row[cat]) * score
            sum += addval
        return sum

    def calculate_points(self, 
                    fpts_dict : Dict[str, pd.DataFrame], 
                    pfref_colNames : Dict[str, str] = PFREF_COL_NAMES):
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
        df['Player'] = df['Player'].str.replace('[\*\+]', '', regex=True).str.strip()

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
        
    def setDefaultPrepSteps(self, prepSteps : List[PreparationStep]):
        if prepSteps:
            return prepSteps
        return []
    
    @abstractmethod
    def loadData():
        pass

    def performSteps(self) -> pd.DataFrame:
        print("\nLoading data from source...")
        df = self.loadData()
        for step in self.prepSteps:
            print(f"Performing step {step.name}")
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

    def setDefaultPrepSteps(self, prepSteps : List[PreparationStep]):
        if prepSteps:
            super().setDefaultPrepSteps(prepSteps)
        prepSteps = [PreparationStep('Get Previous Year Data', self._createPreviousYear)]
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

    def _addCurrentRosters(self, df : pd.DataFrame):
        # Not used currently
        '''
        cols = list(df.columns) + ['rookie', 'draftPick']
        # Merge by pfref_id, fill in other info after
        merged = df.merge(self.currentRosterDf, on = ['pfref_id','Year'], how = 'outer')
        for i in ['Player', 'Tm', 'FantPos']:
            merged[i] = merged[i + '_x'].fillna(merged[i + '_y'])
        final_df = merged[cols].copy()
        # Re-do changedTeam var for 2023 guys
        final_df['changedTeam'] = np.where((final_df['Tm'] == merged['PrvTm']) | (merged['PrvTm'].isnull()), 0, 1)
        return final_df
        '''
        
        
    def _createPreviousYear(self, pts_df_base : pd.DataFrame) -> pd.DataFrame:
        scoring_var = self.scoringType.points_name()
        # 1. Create template
        predTemplate = pts_df_base[['Player', 'Tm', 'Age', 'FantPos', 'Year', 'pfref_id', scoring_var]]
        print(len(pts_df_base[pts_df_base['Year'] == 2022]))
        
        print("1. This is the shape of og dataset...")
        print(predTemplate)
        print(predTemplate.shape)
        
        # 2. Merge on last year's results
        # Change to only merge by pfref_id
        # Add back ['Player', 'Tm', 'FantPos] via roster
        prvYr = predTemplate[['Tm', 'Age', 'Year', 'pfref_id', scoring_var]].copy()
        prvYr.rename(columns = {'Year' : 'PrvYear', 
                                'Tm' : 'PrvTm', 
                                scoring_var : 'Prv' + scoring_var, 
                                'Age' : 'PrvAge'}, 
                                inplace = True)
        prvYr['Year'] = prvYr['PrvYear'] + 1
        prvYr['PrvAge'] = pd.to_numeric(prvYr['PrvAge'])
        merged = predTemplate.merge(prvYr, on = ['pfref_id', 'Year'], how = 'outer', indicator= 'foundLastYearStats')
        
        print("2. This is the shape after doing outer join with previous years' data")
        print(merged.shape)
        print(merged['foundLastYearStats'].value_counts())

        # 3. Remove right_only obs that aren't from 2022
        print(f"3. In total, {len(merged[merged['foundLastYearStats'] == 'right_only'])} observations are players with some data, "
              + " but no previous year statistics")
        print(f"-Of these, {len(merged[merged['PrvYear'] == (self.currentYear - 1)])} observations are associated" 
              + "with year {self.currentYear}")
        print("\t-These observations don't have a 'y-value' for regression, but this is the year we are trying to predict, so OK")
        print(f"-Remove remaining {len(merged[(merged['PrvYear'] != (self.currentYear - 1)) & (merged['foundLastYearStats'] == 'right_only')])} observations")
        print("\t-These observations don't have a 'y-value' for regression, only 'x-values', so ok to delete")
        merged = merged[(merged['Year'] == self.currentYear) | (merged['foundLastYearStats'] != 'right_only')]
        print(merged[merged['foundLastYearStats'] == 'left_only'])   
        print(merged.shape)
        print()
        
        # 4. Create found last year flag
        # This will help distinguish rookies and other players not in data
        merged['PrvYear'] = merged['Year'] - 1

        # 5. Examine composition of remaining observations  
        # Left_only and both are needed for regression - excludes 2013 observations
        # Right_only needed for prediction - excludes non-2022 observations (right-only's in OG data)
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
        return merged.drop('foundLastYearStats', axis = 1)
    
    def _create_qb_chg(self, df : pd.DataFrame) -> pd.DataFrame:
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
        
    def _getPtShare(self, df : pd.DataFrame):
        scoring_var : str = self.scoringType.points_name()
        prv_scoring_var = 'Prv' + scoring_var
        df['ones'] = 1
        df_tm = df.groupby(['Tm','FantPos','Year'], as_index = False).sum()[['Tm','FantPos','Year', prv_scoring_var, 'ones']]
        df_tm = df_tm.rename(columns = {prv_scoring_var : 'PrvYrTmPts', 'ones' : 'PlayersAtPosition'})
        df_tm = df_tm[df_tm['Tm'].str[-2:] != 'TM']
        
        df = df.merge(df_tm, on = ['Tm','FantPos','Year'], how = 'left')
        df['PrvYrPtsShare'] = df[prv_scoring_var] / df['PrvYrTmPts'] 
        # Keeping in, because only real conflict with PrvYrTmPts and PrvPts_PPR is multicollinearity
        # df.loc[df['PrvYrPtsShare'].isnull(), 'PrvYrPtsShare'] = 1 / df.loc[df['PrvYrPtsShare'].isnull(), 'PlayersAtPosition']
        df.drop('ones', axis = 1, inplace = True)
        return df


if __name__ == '__main__':
    # pd.options.display.max_columns = None
    path = pathlib.Path(__file__).parent.resolve()
    os.chdir(path)

    SCORING = ScoringType.HPPR

    # =======
    # Points
    # =======
    pc = PointsConverter(SCORING)
    points_sources = ['../../data/created/points.p']
    pointsDataset = PointsDataset(points_sources, SCORING, pc)
    pt = pointsDataset.performSteps()
    print(pt)