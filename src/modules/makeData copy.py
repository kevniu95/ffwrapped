# from .draft import *
# from typing import Set
from itertools import product
import time
# import joblib
import logging
import string
import pandas as pd

from ..util.logger_config import setup_logger
LOG_LEVEL = logging.INFO
logger = setup_logger(__name__, level = logging.INFO)

def get_nth_best_players(group) -> pd.DataFrame:
    pos_counts = {'QB' : 5, 'RB' : 10, 'WR' : 10, 'TE' : 5}
    selected_players = []
    for pos, n in pos_counts.items():
        # Get the nth best player for each position
        pos_group = group[group['FantPos'] == pos].reset_index()
        nth_best = pos_group.iloc[n-1 : n]
        selected_players.append(nth_best)
    return pd.concat(selected_players)

def getBaselinePointDict(undrafted: pd.DataFrame, points_name: str) -> Dict[str, float]:
    undrafted = undrafted.sort_values(by=['FantPos', points_name], ascending=[True, False])
    baseline_points_df = undrafted[['FantPos','team','pred', points_name]].groupby(['FantPos'])[['FantPos','team','pred',points_name]].apply(get_nth_best_players).reset_index(drop=True)
    return dict(zip(baseline_points_df['FantPos'], baseline_points_df[points_name]))

def generateBaselines(baselines_points : Dict[str, float], teams : Set[Team], positions = ['QB','RB','WR','TE']) -> pd.DataFrame:
    weeks = range(1, 19)
    baseline_rows = list(product(weeks, teams, positions))

    baseline_data = {'Week': [row[0] for row in baseline_rows],
                    'team': [row[1].id for row in baseline_rows],
                    'FantPos': [row[2] for row in baseline_rows],
                    'PPR': [baselines_points[row[2]] / 18 for row in baseline_rows],
                    'Player' : [f'baseline{row[2]}' for row in baseline_rows]
    }
    return pd.DataFrame(baseline_data)

def finalizeWeeklyDf(drafted: pd.DataFrame, appendMe: pd.DataFrame, scoringType: ScoringType, year: int) -> pd.DataFrame:
    points_name = scoringType.points_name()
    df_dict = pd.read_pickle(f'../../data/imports/created/weekly_points/weekly_points_{year}.p')
    weekly_df  = pd.concat(df_dict.values())
    weekly_df = weekly_df.merge(drafted[['pfref_id','team', 'FantPos']], on = 'pfref_id')
    weekly_df = pd.concat([weekly_df, appendMe, appendMe]) # At max, 2 waiver-wire baselines can be selected starters per position

    weekly_df[['Week','FantPt','Rec']] = weekly_df[['Week','FantPt','Rec']].apply(pd.to_numeric, errors = 'coerce').fillna(0)
    weekly_df[points_name] = weekly_df['FantPt'] + (scoringType.value * weekly_df['Rec'])
    return weekly_df

def generateY(drafted : pd.DataFrame, appendMe : pd.DataFrame, scoringType : ScoringType, year: int):
    points_name = scoringType.points_name()
    logger.debug("Generating weekly starters and aggregating...")
    st = time.time()
    weekly_df = finalizeWeeklyDf(drafted, appendMe, scoringType, year)
    # logger.info(f"Time to finalize weekly df: {time.time() - st}")
    # Get non-FLEX starters
    weekly_df.sort_values(['Week','team', 'FantPos', points_name], ascending = [True, True, True, False], inplace = True)
    weekly_df.reset_index(drop = True, inplace = True)
    weekly_df[points_name] = pd.to_numeric(weekly_df[points_name], errors = 'coerce')
    # print(weekly_df[weekly_df[points_name].isnull()])
    weekly_df[points_name] = weekly_df[points_name].astype(float)
    weekly_df['Rank'] = weekly_df[['Week','team','FantPos', points_name]].groupby(['Week','team','FantPos']).rank(ascending = False, method = 'first')[[points_name]]
    pos_counts = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1}
    weekly_df['pos_starters'] = weekly_df['FantPos'].map(pos_counts).astype(int)
    weekly_df_starters = weekly_df.loc[weekly_df['Rank'] <= weekly_df['pos_starters']]
    
    # # Split rest, and find best FLEX
    weekly_df_flex_cands = weekly_df[(~weekly_df.index.isin(weekly_df_starters.index)) & (weekly_df['FantPos'].isin(['TE','WR','RB']))].copy()
    weekly_df_flex_cands['FantPos'] = 'FLEX'
    weekly_df_flex_cands.sort_values(['Week','team',points_name],ascending = [True, True, False], inplace= True)
    weekly_flex_starters = weekly_df_flex_cands.groupby(['Week','team','FantPos'], as_index = False).first()
    starting_lineup = pd.concat([weekly_df_starters, weekly_flex_starters])
    # starting_lineup = weekly_df.groupby(['Week', 'team']).apply(getWeeklyHigh).reset_index(drop=True)

    y = starting_lineup[['team','FantPos',points_name]].groupby(['team','FantPos'], as_index = False).sum()[['team','FantPos', points_name]]
    y[points_name] = y[points_name] / 18
    return y

def getRosterConfigVariables(df : pd.DataFrame, league : League) -> pd.DataFrame:
    lst = []
    df.sort_values(['team', 'FantPos', 'pred'], ascending = [True, True, False], inplace = True)
    for team in df['team'].unique():
        team = league.getTeamFromId(team)
        rows = team.getRosterConfigOneTeam(df)
        lst.extend(rows)
        # new_df = pd.concat([new_df, pd.DataFrame([row])], ignore_index=True)
    df_extended = pd.DataFrame(lst, columns=rows[0].keys())
    return df_extended

def makeData(year : int, temp : float, scoringType = ScoringType, leagueId : int = None, savePlayerList: bool = True) -> pd.DataFrame:
    if not leagueId:
        alphabet = string.ascii_lowercase + string.digits
        leagueId = ''.join(random.choices(alphabet, k=8))

    # Simulate draft
    logger.debug(f"Simulating draft for year {year}...")
    finishedDraft = simulateLeagueAndDraft(year, temp, scoringType = scoringType)
    df = finishedDraft.pool.df
    undrafted = df.loc[df['team'].isnull()]
    drafted = df.loc[df['team'].notnull(), ['pfref_id','team', 'FantPos', 'pred', 'var_pred']]
    
    # Undrafted Players -> baseline
    points_name = scoringType.points_name()
    logger.debug(f"Generating baselines...")
    # st = time.time()
    if year != 2023:
        baselinePoints = getBaselinePointDict(undrafted, points_name)
        appendMe : pd.DataFrame = generateBaselines(baselinePoints, finishedDraft.league.teams)
    # logger.info(f"Time to generate baselines: {time.time() - st}")

    # Generate Y - Average Total PF for each team, for each position
        # Total by week -  so 2x average RB score for RB
    logger.debug("Generating y...")
    # if year != 2023:
    # st = time.time()
    y = generateY(drafted, appendMe, scoringType, year)
    # logger.info(f"Time to generate y: {time.time() - st}")
    # else:
    # y = drafted[['team', 'FantPos']].drop_duplicates()
    # y[points_name] = np.nan
    # y.sort_values(['team','FantPos'], inplace = True)
    # st = time.time()
    x = getRosterConfigVariables(drafted, finishedDraft.league)
    # logger.info(f"Time to generate x: {time.time() - st}")

    # st = time.time()
    x.sort_values(['team','FantPos'], inplace= True)
    z = y.merge(x, on = ['team', 'FantPos'], how = 'right')
    z['league'] = leagueId
    z['year'] = year
    # logger.info(f"Time to merge x and y: {time.time() - st}")

    # score_dfs = []
    # for team in x['team'].unique():
    #     sub_df = x[x['team'] == team].copy()
    #     pred_score = getTeamScoreFromRosterConfig(sub_df, models)
    #     pred_score['team'] = team
    #     score_dfs.append(pred_score[['team','Total']])
    # score_df = pd.concat(score_dfs)
    
    # if savePlayerList:
    #     df['league'] = leagueId
    #     keep_fields = ['league','Player', 'pfref_id','team', 'FantPos', scoringType.adp_column_name(), 'pred', 'var_pred', 'pick']
    #     save_drafted = df.loc[df['team'].notnull(), keep_fields]

    #     picks = save_drafted[['league','team','pick','pfref_id', 'FantPos']]
    #     final_picks = score_df[['team','Total']].merge(picks)
    #     final_picks = final_picks[['league','team','Total','pick','pfref_id','FantPos']].sort_values(['team','pick'])
        
    #     path = f'../../data/research/created/regression/picks2023Draft_f.csv'
    #     if os.path.exists(path):
    #         final_picks.to_csv(path, mode = 'a', header = False, index = False)
    #     else:
    #         final_picks.to_csv(path, index = False)
        
    return z[['year','league','team','FantPos',points_name, 'A_pred_points','A_std_error', 'B_pred_points', 'B_std_error', 'num_others', 'avg_pred_points_other', 'avg_std_error_other']]

# def getTeamScoreFromRosterConfig(df : pd.DataFrame, models : Dict[str, sklearn.pipeline.Pipeline]) -> pd.DataFrame:
#     df_extended = df
#     sum = 0
#     score_dict = {}
#     for pos in ['QB', 'RB', 'WR', 'TE', 'FLEX']:
#         base_vars = ['A_pred_points',
#             'A_std_error',
#             'num_others',
#             'avg_pred_points_other',
#             'avg_std_error_other'
#             ]
#         model = models[pos]
#         if pos in['RB','WR']:
#             base_vars = base_vars + ['B_pred_points', 'B_std_error']

#         base = model.predict(df_extended.loc[df_extended['FantPos'] == pos, base_vars])[0]
#         score_dict[pos] = base
#         sum += base
#     score_dict['Total'] = sum
#     return pd.DataFrame([score_dict])

# def initializeModels(path = f'../../data/research/created/results') -> Dict[str, sklearn.pipeline.Pipeline]:
#     positions = ['QB','RB','TE','WR','FLEX']
#     models = {}

#     for pos in positions:
#         file_path = path + f'/{pos}_rosterconfigreg_params_1.joblib'
#         model = joblib.load(file_path)
#         models[pos] = model
#     return models    

if __name__ == '__main__':
    pd.options.display.max_columns = None
    
    path = pathlib.Path(__file__).parent.resolve()
    os.chdir(path)

    # print(makeData(2016, 4, ScoringType.HPPR))
    for i in range(2500):
        st = time.time()
        if i % 50 == 0:
            logger.info(f"Starting iteration {i}")
            logger.info(f"10 its took {time.time()  - st}")
            st = time.time()
        for year in range(2016, 2023):
            # print(year)
            # time how long it takes to run this funciton
            # start_time = time.time()
            a = makeData(year, 4, ScoringType.HPPR)
            # end_time = time.time()
            # print(f"Time to run: {end_time - start_time}")
            path = f'../../data/regression/rosterConfig/rosterConfigData.csv'
            if os.path.exists(path):
                a.to_csv(path, mode = 'a', header = False, index = False)
            else:
                a.to_csv(path, index = False)
            
    # f'../../data/regression/rosterConfig/test.csv'
    # a.to_csv(f'../../data/regression/rosterConfig/test.csv', index = False)
    # models = initializeModels()

    # for year in range(2022, 2023):
    #     print(year)
    #     for i in range(1):
    #         print(i)
    #         df = makeData(year, 4, i + 1, ScoringType.HPPR)
    #         # path = f'../../data/research/created/regression/maxByRosterConfig_2.csv'
    #         # if os.path.exists(path):
    #         #     df.to_csv(path, mode = 'a', header = False, index = False)
    #         # else:
    #         #     df.to_csv(path, index = False)
    #         print()
    
    # for i in range(7500):
    #     print(i)
    #     df = makeData(2023, 3, models, ScoringType.HPPR)
    #     path = f'../../data/research/created/regression/rosterConfig2023Draft_f.csv'
    #     print(df.shape)
    #     if os.path.exists(path):
    #         df.to_csv(path, mode = 'a', header = False, index = False)
    #     else:
    #         df.to_csv(path, index = False)
    
    # for i in range(2500):
    #     print(i)
    #     df = makeData(2023, 2.5, models, ScoringType.HPPR)
    #     path = f'../../data/research/created/regression/rosterConfig2023Draft_f.csv'
    #     print(df.shape)
    #     if os.path.exists(path):
    #         df.to_csv(path, mode = 'a', header = False, index = False)
    #     else:
    #         df.to_csv(path, index = False)
        