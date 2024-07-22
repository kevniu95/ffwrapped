import pandas as pd
from .doPrep import *
from .regression import *
import pathlib
import os
import joblib
import time
import re

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
pd.set_option('display.max_columns', None)

ROSTER_CONFIG_REGRESSION_DATA = '../../data/regression/rosterConfig/rosterConfigData1.csv'
scoringType = ScoringType.HPPR

def uploadSource(path: str = ROSTER_CONFIG_REGRESSION_DATA) -> pd.DataFrame:
    base_df = pd.read_csv(path).reset_index()
    bad_leagues = base_df.loc[base_df['num_others'] == -1, 'league'].drop_duplicates()
    logger.info("Number of leagues dropped is %s", len(bad_leagues))
    base_df = base_df.loc[~base_df['league'].isin(bad_leagues)]
    return base_df


def writeDfRowToFile(results: pd.DataFrame, 
                     results_path: str = '../../data/regression/rosterConfig/allResults.csv'
                     ) -> None:
    file_exists = os.path.isfile(results_path)
    is_empty = not os.path.getsize(results_path) > 0 if file_exists else False    
    # Write to CSV, adding header if file doesn't exist or is empty
    results.to_csv(results_path, mode='a', index=False, header=not file_exists or is_empty)
    
    
def createDfRow(name : str,
                 description: str,
                 model: str,
                 time_to_finish: float,
                 train_r2: float,
                 test_r2: float
                 ) -> pd.DataFrame:
    return pd.DataFrame({'Name': [name],
                          'Description': [description],
                          'Model': model, 
                          'TimeToFinish': [time_to_finish],
                          'TrainR2': [train_r2],
                          'TestR2': [test_r2]
                      })


def getModelName(model: sklearn.pipeline.Pipeline,
                 polys: int = None,
                 shortVersion: bool = True) -> str:
    model_name = str(model) 
    if polys:
        model_name += '_poly' + str(polys)
    if shortVersion:
        model_name = re.sub(r"\s+", " ", model_name)
        model_name = re.sub(r"\(.*\)", "", model_name)
    return model_name
    

# =====================
# Starter
# =====================
def _mapRosterConfigToStarter(df: pd.DataFrame) -> pd.DataFrame:
    df1 = df[['year', 'league', 'team', scoringType.points_name(), 'A_pred_points']].groupby(['year','league','team']).sum([scoringType.points_name(), 'A_pred_points']).reset_index()
    df2 = df.loc[df['FantPos'].isin(['RB','WR']), ['year', 'league', 'team','B_pred_points']].groupby(['year','league','team']).sum('B_pred_points').reset_index()
    df3 = df1.merge(df2)
    df3['final_pred_points'] = df3['A_pred_points'] + df3['B_pred_points']
    return df3[['team', scoringType.points_name(), 'final_pred_points']]

def _prepStarterLinearRegresion(path: str = ROSTER_CONFIG_REGRESSION_DATA) -> pd.DataFrame:
    return _mapRosterConfigToStarter(uploadSource(path))
                        
def doRosterConfigStarter(save_coefs: bool = True,
                          save_coefs_path: str = '../../data/regression/rosterConfig/parameters/{}.joblib',
                          results_path: str = '../../data/regression/rosterConfig/allResults.csv',
                          model: sklearn.pipeline.Pipeline = LASSO_CV_REGRESSOR) -> None:
    name = "Starter"
    x_description = '''One variable: total final predicted points of starters'''
    
    df = _prepStarterLinearRegresion()
    for poly in [1, 2, 3]:
        model_res, train_r2, test_r2, time_to_finish = split_and_try_model(df, scoringType.points_name(), ['final_pred_points'], polys = poly, regressor = model)
        model_name = getModelName(model, poly, shortVersion=False)
        df_row = createDfRow(name, x_description, model_name, time_to_finish, train_r2, test_r2)
        writeDfRowToFile(df_row, results_path)

        if save_coefs: 
            clean_model_name = getModelName(model, poly)
            results_file_name = name.lower() + '_' + clean_model_name
            joblib.dump(model_res, save_coefs_path.format(results_file_name))
        
# =====================
# One Line
# =====================
def _prepOneLineDataFrame() -> Tuple[pd.DataFrame, List[str]]:
    og_df = uploadSource()
    df = og_df[og_df['FantPos'].isin(['QB','RB','WR','TE'])].copy()

    column_names = ["A_pred_points",
                    "A_std_error",
                    "B_pred_points",
                    "B_std_error",
                    "num_others",
                    "avg_pred_points_other",
                    "avg_std_error_other"
                    ]
    
    pivot_df = df.pivot_table(index=['year', 'league', 'team'], 
                          columns='FantPos', 
                          values=column_names,
                          aggfunc='first')
    pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]
    pivot_df.reset_index(inplace=True)
    pivot_df.drop(columns = ['B_pred_points_QB', 'B_pred_points_TE', 'B_std_error_QB', 'B_std_error_TE'], inplace = True)

    y = og_df[['year','league','team',scoringType.points_name()]].groupby(['year','league','team']).sum().reset_index()
    pivot_df = pivot_df.merge(y, on = ['year','league','team'])

    col_list = list(pivot_df)[3:-1]
    return pivot_df, col_list

def doRosterConfigOneLine(save_coefs:bool = True,
                          save_coefs_path: str = '../../data/regression/rosterConfig/parameters/{}.joblib',
                          results_path: str = '../../data/regression/rosterConfig/allResults.csv',
                          model: sklearn.pipeline.Pipeline = LASSO_CV_REGRESSOR) -> None:        
    name = "OneLine"
    x_description = """Each team is represented by one single row of all roster config variables"""
    
    df, col_list = _prepOneLineDataFrame()
    for poly in [1, 2]:
        model_res, train_r2, test_r2, time_to_finish = split_and_try_model(df, scoringType.points_name(), col_list, polys = poly, regressor = model)
        model_name = getModelName(model, poly, shortVersion=False)
        df_row = createDfRow(name, x_description, model_name, time_to_finish, train_r2, test_r2)
        writeDfRowToFile(df_row, results_path)

        if save_coefs: 
            clean_model_name = getModelName(model, poly)
            results_file_name = name.lower() + '_' + clean_model_name
            joblib.dump(model_res, save_coefs_path.format(results_file_name))             

# =====================
# Simple
# =====================
def _prepSimple() -> Tuple[pd.DataFrame, List[str]]:
    og_df = uploadSource()
    df = og_df[og_df['FantPos'].isin(['QB','RB','WR','TE'])].copy()
    df['total_points']  = df['num_others'] * df['avg_pred_points_other'] + df['A_pred_points'] + df['B_pred_points']
    df['total_players'] = df['num_others'] + 1 + df['B_pred_points'].apply(lambda x: 1 if x > 0 else 0)
    df['average_points'] = df['total_points'] / df['total_players']
    df['position_variance'] = (df['num_others'] * (df['avg_std_error_other'] ** 2)) + df['A_std_error'] ** 2 + df['B_std_error'].replace(10000, 0) ** 2
    df['std_error'] = df['position_variance'] ** 0.5
    
    pivot_df = df.pivot_table(index=['year', 'league', 'team'], 
                          columns='FantPos', 
                          values=['average_points', 'total_players', 'std_error'], 
                          aggfunc='first')
    pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]
    pivot_df.reset_index(inplace=True)
    
    y = og_df[['year','league','team',scoringType.points_name()]].groupby(['year','league','team']).sum().reset_index()
    pivot_df = pivot_df.merge(y, on = ['year','league','team'])

    x_values = []
    for pos in ['QB','RB','WR','TE']:
        for field in ['average_points', 'total_players', 'std_error']:
            x_values.append(f'{field}_{pos}')
    return pivot_df, x_values
    
def doRosterConfigSimple(save_coefs:bool = True,
                          save_coefs_path: str = '../../data/regression/rosterConfig/parameters/{}.joblib',
                          results_path: str = '../../data/regression/rosterConfig/allResults.csv',
                          model: sklearn.pipeline.Pipeline = LASSO_CV_REGRESSOR) -> None:        
    name = "Simple"
    x_description = """Each team is represented average and total variance at each position"""
    
    df, col_list = _prepSimple()
    for poly in [1, 2]:
        model_res, train_r2, test_r2, time_to_finish = split_and_try_model(df, scoringType.points_name(), col_list, polys = poly, regressor = model)
        model_name = getModelName(model, poly, shortVersion=False)
        df_row = createDfRow(name, x_description, model_name, time_to_finish, train_r2, test_r2)
        writeDfRowToFile(df_row, results_path)

        if save_coefs: 
            clean_model_name = getModelName(model, poly)
            results_file_name = name.lower() + '_' + clean_model_name
            joblib.dump(model_res, save_coefs_path.format(results_file_name))             







# ==================
# By position
# ==================

def doRosterConfigLinearRegression(save: bool = True):
    df = uploadSource()
    for position in ['QB','RB','WR','TE','FLEX']:
        print('====================')
        print(position)
        print('====================')
        df_sub = df.loc[df['FantPos'] == position].copy()
        # best_model = None
        # best_model_r2 = 0
        for poly in [1]:
            print(f'Polys: {poly}')
            model_results = split_and_try_model(df_sub, scoringType.points_name(), ['A_pred_points'], polys = poly)
            # print(model_results)
            # if model_results[2] > best_model_r2:
            # best_model_r2 = model_results[1]
            best_model = model_results[0]
        if save:
            print('dumping results...')
            path_name = f'../../data/regression/rosterConfig/results/linear_{position}_rosterconfigreg_params.joblib'
            joblib.dump(best_model, path_name)
        
        df_sub['pred'] = best_model.predict(df_sub[['A_pred_points']])
        df.loc[df_sub['index'], 'pred'] = df_sub['pred']
        print(df_sub[['year','league','team',scoringType.points_name(), 'pred']].head(25))
    # print(df.head())
    ret_df = df[['year','league', 'FantPos','team',scoringType.points_name(), 'pred']]
    print(ret_df.head(25))
    a = ret_df.groupby(['year','league','team']).sum().reset_index()
    print(r2_score(a[scoringType.points_name()], a['pred']))
    print(a.head())
    return a

def doRosterConfigNNRegression():
    pass

def evaluateResults():
    pass

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

def doRegressionOnPosition(position : str, 
                           base_df : pd.DataFrame, 
                           scoringType : ScoringType, 
                           save : bool = True, 
                           points_only : bool = False, 
                           polys : int = 3, 
                           regressor = LASSO_CV_REGRESSOR) -> None:
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

def initializeModels(path = f'../../data/regression/rosterConfig/results') -> Dict[str, sklearn.pipeline.Pipeline]:
    positions = ['QB','RB','TE','WR','FLEX']
    models = {}

    for pos in positions:
        file_path = path + f'/{pos}_rosterconfigreg_params.joblib'
        model = joblib.load(file_path)
        models[pos] = model
    return models   

if __name__ == '__main__':
    path = pathlib.Path(__file__).parent.resolve()
    os.chdir(path)
    

    standard_nn_model = MLPRegressor(max_iter=300, hidden_layer_sizes=(100, 100), activation='tanh', solver='adam', alpha=0.005, learning_rate_init= 0.0005)
    calls = [doRosterConfigStarter, doRosterConfigOneLine, doRosterConfigSimple]
    for call in calls:
        try:
            logger.info(f"Running {call}")
            call()
        except Exception as e:
            logger.error(f"Error in {call}: {e}")
        try:
            logger.info(f"Running {call} with Neural Network")
            call(model = standard_nn_model)
        except Exception as e:
            logger.error(f"Error in {call}: {e}")

    # doRosterConfigOneLine(model = standard_nn_model)
    # 2024-07-21 14:17:29,094 - INFO - regression - try_model - Finished fitting model in 731.7506449222565 seconds
    # R^2 score on training data 0.4260044473278135
    # R^2 score on test data 0.3942941820678214

    # doRegressions(ScoringType.HPPR, points_only = False)
    # models = initializeModels()
    # print(models)
    # doRosterConfigStarter()
    # doRosterConfigLinearRegression()
    # doRosterConfigOneLineRegression()
    # doRegressionOnPosition('QB', )
    # def doRegressionOnPosition(position : str, base_df : pd.DataFrame, scoringType : ScoringType, save : bool = True, points_only : bool = False, polys : int = 3, regressor = LASSO_CV_REGRESSOR) -> None:
    # doRegressions(ScoringType.HPPR, points_only = False)
    # doPivotRegression(polys = 1, regressor = MLPRegressor(hidden_layer_sizes = (50, 50), solver = 'adam', activation = 'tanh', max_iter = 400))