import pathlib
import os
import re
from .doImport import *
from .doPrep import *
from .regression import *
from ..domain.common import ScoringType, thisFootballYear

LAST_EXCLUDE_YEAR = 2015
ADJUSTED_PREDICT_YEAR = thisFootballYear()  - LAST_EXCLUDE_YEAR
LOG_LEVEL = logging.INFO
logger = setup_logger(__name__, level = LOG_LEVEL)

# =================
# Gets ADP Info on
# =================
def _getPtShare(df : pd.DataFrame, scoringType : ScoringType) -> pd.DataFrame:
    # Need to calculate these values for new players
    # Just like we did for prvYear and other variables before this call
    df.drop('PrvYrTmPtsAtPosition', axis = 1, inplace = True)
    df.drop('PlayersAtPosition', axis = 1, inplace = True)
    scoring_var : str = scoringType.points_name()
    prv_scoring_var = 'Prv' + scoring_var
    df['ones'] = 1
    df_tm = df[['Tm','FantPos','Year', prv_scoring_var, 'ones']].groupby(['Tm','FantPos','Year'], as_index = False).sum()[['Tm','FantPos','Year', prv_scoring_var, 'ones']]
    df_tm = df_tm.rename(columns = {prv_scoring_var : 'PrvYrTmPtsAtPosition', 'ones' : 'PlayersAtPosition'})
    df_tm = df_tm[df_tm['Tm'].str[-2:] != 'TM']
    
    df = df.merge(df_tm, on = ['Tm','FantPos','Year'], how = 'left')
    df['PrvYrPtsShare'] = df[prv_scoring_var] / df['PrvYrTmPtsAtPosition'] 
    # Keeping in, because only real conflict with PrvYrTmPts and PrvPts_PPR is multicollinearity
    # df.loc[df['PrvYrPtsShare'].isnull(), 'PrvYrPtsShare'] = 1 / df.loc[df['PrvYrPtsShare'].isnull(), 'PlayersAtPosition']
    df.drop('ones', axis = 1, inplace = True)
    return df

def mergeAdpToPoints(pts_df_reg: pd.DataFrame, adp_df: pd.DataFrame, scoringType : ScoringType) -> pd.DataFrame:
    logger.info("Merging ADP Dataset to Points Dataset" )
    # Objective of method: update player info on points side
    # pts_df_reg.loc[(pts_df_reg['Player'] == 'Terrelle Pryor') & (pts_df_reg['Age'] >= 26), 'FantPos'] = 'WR'
    pts_df_reg.loc[(pts_df_reg['Player'] == 'Danny Woodhead'), 'FantPos'] = 'RB'
    pts_df_reg.loc[(pts_df_reg['Player'] == 'Ladarius Green'), 'FantPos'] = 'TE'

    # i. Update player names for merge
    adp_df.loc[adp_df['Name'] == 'Robbie Anderson', 'Name'] = 'Chosen Anderson'
    adp_df.loc[adp_df['Name'] == 'Travis Etienne Jr.', 'Name'] = 'Travis Etienne'
    adp_df.loc[adp_df['Name'] == 'Joshua Palmer', 'Name'] = 'Josh Palmer'
    adp_df.loc[adp_df['Name'] == 'Jeff Wilson Jr.', 'Name'] = 'Jeff Wilson'
    adp_df.loc[adp_df['Name'] == 'J.J. Nelson', 'Name'] = 'JJ Nelson'
    adp_df.loc[(adp_df['Name'] == 'Cordarrelle Patterson') & (adp_df['Year'] < 2021), 'Position'] = 'WR'
    adp_df.loc[(adp_df['Name'] == 'Ty Montgomery') & (adp_df['Year'] .isin([2016, 2017, 2018])), 'Position'] = 'RB'
    adp_df.loc[adp_df['Name'] == 'Brian Robinson', 'Name'] = 'Brian Robinson Jr.'
    adp_df.loc[adp_df['Name'] == 'Scotty Miller', 'Name'] = 'Scott Miller'

    # 1. Merge with adp info
    merged = pts_df_reg.merge(adp_df[['Name', 'Year', 'Team', 'Position', scoringType.adp_column_name()]],
                        left_on = ['Player','Year','FantPos'], 
                        right_on = ['Name', 'Year', 'Position'],
                        how = 'outer',
                        indicator= 'foundAdp').copy()
    
    # 2. Create previous year, fill out player name, position, team
    # Filling data where only ADP info was available 
    merged['PrvYear'] = merged['Year'] - 1
    merged.fillna({'Player' : merged['Name'], 
                   'FantPos' : merged['Position'], 
                   'Tm' : merged['Team']}, inplace = True)
    merged.drop(['Name','Position','Team'], axis = 1, inplace = True)

    # 3. Create positional dummies
    merged[['QB','RB','TE','WR']] = pd.get_dummies(merged['FantPos'])
    merged = _getPtShare(merged, scoringType)
    merged['drafted'] = np.where(merged['foundAdp'] == 'left_only', False, True)
    merged[scoringType.adp_column_name() + 'Sq'] = merged[scoringType.adp_column_name()] ** 2
    return merged

# =================
# Do regression
# =================

def makeFinalLimitations(df : pd.DataFrame, 
                         scoring : ScoringType, 
                         lastExcludeYear: int = LAST_EXCLUDE_YEAR,
                         adjustedPredictYear: int = ADJUSTED_PREDICT_YEAR) -> pd.DataFrame:
    # ==========
    # Limitations
    # ==========
    # Remove those with ADP but no stats for that year (really big outliers)
    # Holdouts, big injuries, off-the-field issues
    # e.g., Le'Veon Bell, Ray Rice, Josh Gordon
    df['adjYear'] = df['Year'] - LAST_EXCLUDE_YEAR
    df = df[(df['adjYear'] > 0) & (df['adjYear'] < ADJUSTED_PREDICT_YEAR) & (df['foundAdp']!= 'right_only')
            & (df['Year'] != thisFootballYear())].copy()
    df = df[df[scoring.points_name()].notnull()]
    df = df[df['rookie'].notnull()] # Note this implicitly drops all players without roster info
    df.loc[df['Age'].isnull(), 'Age'] = 23
    return df

def makeDatasetAfterBaseRegression(df : pd.DataFrame, 
                                       scoring : ScoringType, 
                                       save : bool, 
                                       save_path : str = '../../data/regression/reg_w_preds_1.p') -> pd.DataFrame:
    df = makeFinalLimitations(df, scoring)
    base_vars = ['Age', 
                 'adjYear', 
                 'drafted',
                 'PrvYrTmPtsAtPosition',
                 'PlayersAtPosition', 
                  scoring.adp_column_name(), 
                 'rookie', 'Yrs',
                 'QB', 'RB', 'TE', 'WR']
    
    # 1. Predict points
    pts_model_0 = split_and_try_model(df, y_var = scoring.points_name(), x_vars = base_vars, polys = 3, regressor = LASSO_CV_REGRESSOR) 
    df['pred'] = pts_model_0.predict(df[base_vars])
    df['var'] = df[scoring.points_name()] - df['pred']
    df['var2'] = df['var'] **2

    # 2. Predict variance
    base_vars = base_vars + ['pred']
    var_model_0 = split_and_try_model(df, y_var = 'var2', x_vars = base_vars, polys = 2, regressor = LASSO_CV_REGRESSOR) 
    df['var_pred'] = var_model_0.predict(df[base_vars])
    
    # 3. Position check
    pos_sum = df['QB'].astype(int) + df['RB'].astype(int) + df['WR'].astype(int) + df['TE'].astype(int)
    if pos_sum.max() != pos_sum.min():
        raise Exception("Some player has been assigned to too many or too few positions!")
    
    df = df[df[scoring.adp_column_name()] < 350].copy()

    if save:
        # Note: 350 is a cutoff for players who are not drafted
        df = df[df[scoring.adp_column_name()] < 350].copy()
        df.to_pickle(save_path)
    return df

# Load regression results
def loadDatasetAfterBaseRegression(df_path : str = None, use_compressed : bool = True) -> pd.DataFrame:
    path = pathlib.Path(__file__).parent.resolve()
    os.chdir(path)
    if not df_path:
        df_path = '../../data/regression/reg_w_preds_1.p'
        if use_compressed:
            df_path = (re.sub(r'\.p$', '_compressed.p', df_path))        
    return pd.read_pickle(df_path)

def main():
    pd.options.display.max_columns = None
    path = pathlib.Path(__file__).parent.resolve()
    os.chdir(path)

    SCORING = ScoringType.HPPR

    # ======
    # Roster
    # ======
    roster_source = '../../data/imports/created/rosters.p'
    final_roster_df = RosterDataset([roster_source]).performSteps()
    
    # =======
    # Points
    # =======
    pc = PointsConverter(SCORING)
    points_sources = ['../../data/imports/created/points.p']
    final_pts_df = PointsDataset(points_sources, SCORING, pc, currentRosterDf = final_roster_df).performSteps()
    
    # =======
    # ADP
    # =======
    adp_sources = ['../../data/imports/created/adp_full.p',
                   '../../data/imports/created/adp_nppr_full.p']
    final_adp_df = ADPDataset(SCORING, adp_sources).performSteps()
    
    # =======
    # Stage
    # =======
    final_df = mergeAdpToPoints(final_pts_df, final_adp_df, SCORING)
    df = makeDatasetAfterBaseRegression(final_df, SCORING, save = True)
    
if __name__ == '__main__':
    main()
