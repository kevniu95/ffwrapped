import pandas as pd
from .doPrep import *
from .regression import *
import pathlib
import os
import joblib
import time

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

from ..util.logger_config import setup_logger
LOG_LEVEL = logging.INFO
logger = setup_logger(__name__, level = logging.INFO)

# def doPivotRegression(polys : int, regressor : LASSO_CV_REGRESSOR):
#     base_df = pd.read_csv('../../data/research/created/regression/maxByRosterConfig.csv')
#     base_df.reset_index(inplace = True)
#     base_df['team_id'] = base_df['index'] // 5 + 1
#     base_df_ppr = base_df[['index', 'year','league','team_id','team','PPR']].groupby(['year','league','team_id','team']).sum().reset_index()
    
#     fields = ['A_pred_points',
#               'A_std_error',
#               'B_pred_points',
#               'B_std_error',
#               'num_others',
#               'avg_pred_points_other',
#               'avg_std_error_other']
    
#     df_piv = base_df.pivot(index = ['year','league','team_id','team'], columns = 'FantPos', values = fields).reset_index()
#     df_piv.columns = df_piv.columns.map(lambda x: f'{x[1]}{x[0]}')

#     reg_df = base_df_ppr.merge(df_piv, on = ['year', 'league', 'team_id', 'team'])
#     base_vars = [i for i in list(reg_df.columns)[5:] if i not in ['PPR','QBB_pred_points', 'TEB_pred_points', 'QBB_std_error', 'TEB_std_error']]
    
#     adp_pts_model_0 = split_and_try_model(reg_df, 'PPR', base_vars, polys = polys, regressor = regressor)
    
#     reg_df['pred'] = adp_pts_model_0.predict(reg_df[base_vars])
#     print(reg_df[['index','year','league','team_id','team', 'PPR','pred']])
#     return reg_df

def doRegressions(scoringType : ScoringType, points_only : bool = False):
    pos_list = ['QB','RB','WR','TE','FLEX']
    model_list = [MLPRegressor(max_iter=300, hidden_layer_sizes=(100, 100), activation='tanh', solver='adam', alpha=0.001, learning_rate_init= 0.005),
                  MLPRegressor(max_iter=300, hidden_layer_sizes=(100, 100), activation='tanh', solver='adam', alpha=0.001, learning_rate_init= 0.001),
                  MLPRegressor(max_iter=400, hidden_layer_sizes=(100, 100), activation='tanh', solver='adam', alpha=0.001, learning_rate_init= 0.001),
                  MLPRegressor(max_iter=400, hidden_layer_sizes=(100, 100), activation='tanh', solver='adam', alpha=0.001, learning_rate_init= 0.001),
                  MLPRegressor(max_iter=400, hidden_layer_sizes=(150, 150), activation='tanh', solver='adam', alpha=5e-6, learning_rate_init= 0.001)
                  ]
    model_dict = dict(zip(pos_list, model_list))

    base_df = pd.read_csv('../../data/regression/rosterConfig/rosterConfigData.csv')
    base_df1 = pd.read_csv('../../data/regression/rosterConfig/rosterConfigData1.csv')
    base_df = pd.concat([base_df, base_df1])
    base_df.reset_index(inplace = True)
    for position in pos_list:
        print(position)
        reg = model_dict[position]
        base_df = doRegressionOnPosition(position, base_df, scoringType, True, points_only, polys = 3, regressor = reg)
    base_df['team_id'] = base_df['index'] // 5 + 1
    team_res = base_df[['team_id', scoringType.points_name(), 'pred']].groupby(['team_id']).sum().reset_index(drop = True)
    team_res['var'] = (team_res[scoringType.points_name()] - team_res['pred']) **2
    print(team_res)
    print(r2_score(team_res[scoringType.points_name()], team_res['pred']))

def doRegressionOnPosition(position : str, base_df : pd.DataFrame, scoringType : ScoringType, save : bool = True, points_only : bool = False, polys : int = 3, regressor = LASSO_CV_REGRESSOR) -> None:
    print()
    reg_df = base_df.loc[base_df['FantPos'] == position].copy()
    base_vars = ['A_pred_points',
                'A_std_error',
                'num_others',
                'avg_pred_points_other',
                'avg_std_error_other'
                ]
    if position in['RB','WR']:
        base_vars = base_vars + ['B_pred_points', 'B_std_error']
    if points_only:
        base_vars = [i for i in base_vars if 'error' not in i]
    
    st = time.time()
    adp_pts_model_0 = split_and_try_model(reg_df, scoringType.points_name(), base_vars, polys = polys, regressor = regressor)
    logger.info(f"Time to fit model: {time.time() - st}")
    
    if save:
        print('dumping results...')
        path_name = f'../../data/regression/rosterConfig/results/{position}_rosterconfigreg_params.joblib'
        joblib.dump(adp_pts_model_0, path_name)
    
    reg_df['pred'] = adp_pts_model_0.predict(reg_df[base_vars])
    base_df.loc[reg_df['index'], 'pred'] = reg_df['pred']
    return base_df
                
if __name__ == '__main__':
    path = pathlib.Path(__file__).parent.resolve()
    os.chdir(path)
    doRegressions(ScoringType.HPPR, points_only = False)
    # doRegressionOnPosition('QB', )
    # def doRegressionOnPosition(position : str, base_df : pd.DataFrame, scoringType : ScoringType, save : bool = True, points_only : bool = False, polys : int = 3, regressor = LASSO_CV_REGRESSOR) -> None:
    # doRegressions(ScoringType.HPPR, points_only = False)
    # doPivotRegression(polys = 1, regressor = MLPRegressor(hidden_layer_sizes = (50, 50), solver = 'adam', activation = 'tanh', max_iter = 400))