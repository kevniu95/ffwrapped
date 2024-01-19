import pathlib
import os
import re
from .doImport import *
from .doPrep import *


# =================
# Actually merges ADP dataset - this should be updateable
# =================
def makeAdpRegDataset(scoring : ScoringType):
    path = pathlib.Path(__file__).parent.resolve()
    os.chdir(path)
    adp_pickle = '../../data/research/created/adp.p'
    points_pickle = '../../data/research/created/points.p'

    adp_df = pd.read_pickle(adp_pickle)
    with open(points_pickle, 'rb') as handle:
        points_df_dict = pickle.load(handle)

    pc = PointsConverter(ScoringType.PPR)
    points_df = pc.calculate_points(points_df_dict)
    
    return mergeAdpDataset(points_df, adp_df, scoring)

# =================
# Actually do regression 
# NOTE: 8.31 - move this to staging area before real regressions
# =================
def makeDatasetAfterBaseRegression(reg_df : pd.DataFrame, save : bool = False, save_path : str = '../../data/research/created/reg_w_preds.p') -> pd.DataFrame:
    base_vars = ['AverageDraftPositionPPR',
                'AverageDraftPositionPPRSq',
                'drafted',
                # 'draftedBig',
                # 'draftedSmall',
                'QB','RB','TE','WR'
                ]
    reg_df['AverageDraftPositionPPRSq'] = reg_df['AverageDraftPositionPPR'] **2
    reg_df = reg_df[(reg_df['Year'] > 2015) & (reg_df['Pts_PPR'].notnull())].copy()
    reg_df['drafted'] = np.where(reg_df['foundAdp'] == 'left_only', False, True)
    # Points regression
    adp_pts_model_0 = split_and_try_model(reg_df, y_var = 'Pts_PPR', x_vars = base_vars, polys = 2, regressor = LASSO_CV_REGRESSOR) 
    reg_df['pred'] = adp_pts_model_0.predict(reg_df[base_vars])
    reg_df['var'] = reg_df['pred'] - reg_df['Pts_PPR']
    reg_df['var2'] = (reg_df['pred'] - reg_df['Pts_PPR']) ** 2

    # Variance regression
    adp_var_model_0 = split_and_try_model(reg_df, y_var = 'var2', x_vars = base_vars, polys = 2, regressor = LASSO_CV_REGRESSOR)
    reg_df['var_pred'] = adp_var_model_0.predict(reg_df[base_vars])
    
    if save:
        path = pathlib.Path(__file__).parent.resolve()
        os.chdir(path)
        reg_df.to_pickle(save_path)
        reg_df_sub = reg_df[reg_df['AverageDraftPositionPPR'] < 350].copy()
        reg_df_sub.to_pickle(re.sub(r'\.p$', '_compressed.p', save_path))    
    return reg_df

def loadDatasetAfterRegression(df_path : str = None, use_compressed : bool = True) -> pd.DataFrame:
    path = pathlib.Path(__file__).parent.resolve()
    os.chdir(path)
    if not df_path:
        df_path = '../../data/research/created/reg_w_preds_1.p'
        if use_compressed:
            df_path = (re.sub(r'\.p$', '_compressed.p', df_path))        
    return pd.read_pickle(df_path)

# =================
# Gets ADP Info on
# =================
def _getPtShare(df : pd.DataFrame, scoringType : ScoringType):
    # Need to calculate these values for new players
    # Just like we did for prvYear and other variables before this call
    df.drop('PrvYrTmPts', axis = 1, inplace = True)
    df.drop('PlayersAtPosition', axis = 1, inplace = True)
    scoring_var : str = scoringType.points_name()
    prv_scoring_var = 'Prv' + scoring_var
    df['ones'] = 1
    df_tm = df[['Tm','FantPos','Year', prv_scoring_var, 'ones']].groupby(['Tm','FantPos','Year'], as_index = False).sum()[['Tm','FantPos','Year', prv_scoring_var, 'ones']]
    df_tm = df_tm.rename(columns = {prv_scoring_var : 'PrvYrTmPts', 'ones' : 'PlayersAtPosition'})
    df_tm = df_tm[df_tm['Tm'].str[-2:] != 'TM']
    
    df = df.merge(df_tm, on = ['Tm','FantPos','Year'], how = 'left')
    df['PrvYrPtsShare'] = df[prv_scoring_var] / df['PrvYrTmPts'] 
    # Keeping in, because only real conflict with PrvYrTmPts and PrvPts_PPR is multicollinearity
    # df.loc[df['PrvYrPtsShare'].isnull(), 'PrvYrPtsShare'] = 1 / df.loc[df['PrvYrPtsShare'].isnull(), 'PlayersAtPosition']
    df.drop('ones', axis = 1, inplace = True)
    return df

def mergeAdpDataset(pts_df_reg, adp_df, scoringType : ScoringType):
    # Update player info on points side
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
    test = pts_df_reg.merge(adp_df[['Name', 'Year', 'Team', 'Position', scoringType.adp_column_name()]],
                        left_on = ['Player','Year','FantPos'], 
                        right_on = ['Name', 'Year', 'Position'],
                        how = 'outer',
                        indicator= 'foundAdp')
    
    # 2. Create previous year, fill out player name, position, team
    # Filling data where only ADP info was available 
    test['PrvYear'] = test['Year'] - 1
    test['Player'].fillna(test['Name'], inplace=True)
    test.drop('Name',axis = 1, inplace= True)
    test['FantPos'].fillna(test['Position'], inplace=True)
    test.drop('Position',axis = 1, inplace= True)
    test['Tm'].fillna(test['Team'], inplace = True)
    test.drop('Team', axis = 1, inplace= True)

    # 3. Create positional dummies
    test[['QB','RB','TE','WR']] = pd.get_dummies(test['FantPos'])
    test = _getPtShare(test, scoringType)
    # test[scoringType.adp_column_name()] = np.where((test[scoringType.adp_column_name()] > 400) | test[scoringType.adp_column_name()].isnull(), 400, test[scoringType.adp_column_name()])
    return test

def makeDatasetAfterBaseRegression_new(df : pd.DataFrame, scoring : ScoringType, save : bool, save_path : str = '../../data/research/created/reg_w_preds_1.p'):
    # Remove those with ADP but no stats for that year (really big outliers)
    # Holdouts, big injuries, off-the-field issues
    # e.g., Le'Veon Bell, Ray Rice, Josh Gordon
    df['drafted'] = np.where(df['foundAdp'] == 'left_only', False, True)
    df[scoring.adp_column_name() + 'Sq'] = df[scoring.adp_column_name()] **2
    df['adjYear'] = df['Year'] - 2015
    og_df = df.copy()

    # ======
    # ADP-supplement on 2016 on newer data
    # ======
    df = og_df[(og_df['adjYear'] > 0) & (og_df['adjYear'] < 8) & (og_df['foundAdp']!= 'right_only')].copy()
    df.loc[df['Age'].isnull(), 'Age'] = 23
    base_vars = [
                'Age', 
                 'adjYear', 
                'drafted',
                 'PrvYrTmPts',
                 'PlayersAtPosition', 
                 scoring.adp_column_name(), 
                 'QB',
                 'RB',
                 'TE',
                 'WR'
                 ]
    pts_model_0 = split_and_try_model(df, y_var = scoring.points_name(), x_vars = base_vars, polys = 3, regressor = LASSO_CV_REGRESSOR) 
    og_df['pred'] = pts_model_0.predict(og_df[base_vars])
    og_df['var'] = og_df[scoring.points_name()] - og_df['pred']
    og_df['var2'] = og_df['var'] **2
    
    df = og_df[(og_df['adjYear'] > 0) & (og_df['adjYear'] < 8) & (og_df['foundAdp']!= 'right_only')].copy()
    base_vars = [
                'Age', 
                 'adjYear', 
                'drafted',
                 'PrvYrTmPts',
                 scoring.adp_column_name(), 
                 'QB',
                 'RB',
                 'TE',
                 'WR',
                 'pred'
                 ]
    var_model_0 = split_and_try_model(df, y_var = 'var2', x_vars = base_vars, polys = 2, regressor = LASSO_CV_REGRESSOR) 
    og_df['var_pred'] = var_model_0.predict(og_df[base_vars])

    pos_sum = df['QB'].astype(int) + df['RB'].astype(int) + df['WR'].astype(int) + df['TE'].astype(int)
    if pos_sum.max() != pos_sum.min():
        raise Exception("Some player has been assigned to too many or too few positions!")
    
    # print(og_df.loc[og_df['adjYear'] == 8, ['Player', 'FantPos', 'PrvPts_HPPR', 'AverageDraftPositionHPPR','pred','var_pred']].sort_values(scoring.adp_column_name()).head(50))
    og_df.loc[og_df['adjYear'] == 8].to_csv('test_a.csv')
    if save:
        og_df = og_df[og_df[scoring.adp_column_name()] < 350].copy()
        og_df.to_pickle(save_path)
    return df

def main():
    pd.options.display.max_columns = None
    path = pathlib.Path(__file__).parent.resolve()
    os.chdir(path)
    print("hello kevin")

    SCORING = ScoringType.HPPR

    # ======
    # Roster
    # ======
    roster_source = '../../data/created/scraping/rosters2023.csv'
    final_roster_df = RosterDataset([roster_source]).performSteps()
    
    # =======
    # Points
    # =======
    pc = PointsConverter(SCORING)
    points_sources = ['../../data/created/points.p']
    final_pts_df = PointsDataset(points_sources, SCORING, pc, currentRosterDf = final_roster_df).performSteps()
    # print(final_pts_df[final_pts_df['Year'] > 2015].sample(50))
    
    # =======
    # ADP
    # =======
    adp_sources = ['../../data/created/adp_full.p',
                   '../../data/created/adp_nppr_full.p']
    final_adp_df = ADPDataset(SCORING, adp_sources).performSteps()
    # print(final_adp_df.head())
    
    # =======
    # 
    # =======
    final_df = mergeAdpDataset(final_pts_df, final_adp_df, SCORING)
    # TODO: 2024.01.18 Just finished looking at this
    # Pick up from here
    print(final_df.head())
    # a = final_df[final_df['Year'] == 2023]
    # # print(a[a[SCORING.adp_column_name()].notnull()].sort_values(SCORING.adp_column_name()).head(50))
    # makeDatasetAfterBaseRegression_new(final_df, SCORING, save = True)

    # # test = makeAdpRegDataset(ScoringType.PPR)
    # # makeDatasetAfterBaseRegression(reg_df = test, save = False)
    # # print(data_set['Pts_HPPR'].notnull().sum())
    # # print(final_df[final_df['foundAdp'] == 'right_only'].sort_values(SCORING.adp_column_name()).head(50))
    # # print(final_df[(final_df['foundAdp'] == 'left_only') & (final_df['Year'] > 2015)].sort_values(SCORING.points_name(), ascending = False).head(50))
    

if __name__ == '__main__':
    main()

