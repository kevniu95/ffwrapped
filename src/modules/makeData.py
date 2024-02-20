import pathlib
import os
from itertools import product
import time
import logging
import string
import pandas as pd

from .draft import *
from .rosterConfigRegression import initializeModels
from ..domain.common import ScoringType, thisFootballYear, loadDatasetAfterBaseRegression

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
    
    # Get non-FLEX starters
    weekly_df.sort_values(['Week','team', 'FantPos', points_name], ascending = [True, True, True, False], inplace = True)
    weekly_df.reset_index(drop = True, inplace = True)
    weekly_df[points_name] = pd.to_numeric(weekly_df[points_name], errors = 'coerce')
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
    df_extended = pd.DataFrame(lst, columns=rows[0].keys())
    return df_extended

def makeDataForRegression(year : int, temp : float, scoringType = ScoringType, leagueId : int = None, savePlayerList: bool = True) -> pd.DataFrame:
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
    y = generateY(drafted, appendMe, scoringType, year)
    x = getRosterConfigVariables(drafted, finishedDraft.league)
    
    # Merge x and y together
    x.sort_values(['team','FantPos'], inplace= True)
    z = y.merge(x, on = ['team', 'FantPos'], how = 'right')
    z['league'] = leagueId
    z['year'] = year
    return z[['year','league','team','FantPos',points_name, 'A_pred_points','A_std_error', 'B_pred_points', 'B_std_error', 'num_others', 'avg_pred_points_other', 'avg_std_error_other']]

def _finalizePickDataset(df: pd.DataFrame, score_df: pd.DataFrame, leagueId: str):
    df['league'] = leagueId
    
    keep_fields = ['league','team','pick','pfref_id', 'FantPos']
    picks = df.loc[df['team'].notnull(), keep_fields]
    final_picks = score_df[['team','Total']].merge(picks)
    final_picks = final_picks[['league','team','Total','pick','pfref_id','FantPos']].sort_values(['team','pick'])
    return final_picks

def _savePicks(final_picks: pd.DataFrame, year: int, saveLocation: str):
    path = saveLocation.format(year)
    if os.path.exists(path):
        final_picks.to_csv(path, mode = 'a', header = False, index = False)
    else:
        final_picks.to_csv(path, index = False)

def makeDataForQuery(year : int, 
                     temp : float, 
                     scoringType = ScoringType, 
                     models = Dict[str, sklearn.pipeline.Pipeline], 
                     saveLocation = '../../data/regression/queryableDraftPicks{}.csv',
                     leagueId : str = None) -> pd.DataFrame:
    if not leagueId:
        alphabet = string.ascii_lowercase + string.digits
        leagueId = ''.join(random.choices(alphabet, k=8))

    # Simulate draft
    logger.debug(f"Simulating draft for year {year}...")
    finishedDraft = simulateLeagueAndDraft(year, temp, scoringType = scoringType)
    df = finishedDraft.pool.df
    drafted = df.loc[df['team'].notnull(), ['pfref_id','team', 'FantPos', 'pred', 'var_pred']]
    
    # Generate Y - Average Total PF for each team, for each position
        # Total by week -  so 2x average RB score for RB
    y = drafted[['team', 'FantPos']].drop_duplicates()
    y[scoringType.points_name()] = np.nan
    y.sort_values(['team','FantPos'], inplace = True)
    
    x = getRosterConfigVariables(drafted, finishedDraft.league)
    x.sort_values(['team','FantPos'], inplace= True)
    z = y.merge(x, on = ['team', 'FantPos'], how = 'right')
    z['league'] = leagueId
    z['year'] = year
    # print("Done with this iteration.\n")
    
    # Merge x and y together
    x.sort_values(['team','FantPos'], inplace= True)
    z = y.merge(x, on = ['team', 'FantPos'], how = 'right')
    z['league'] = leagueId
    z['year'] = year

    score_dfs = []
    for team in x['team'].unique():
        sub_df = x[x['team'] == team].copy()
        pred_score = getTeamScoreFromRosterConfig(sub_df, models)
        pred_score['team'] = team
        score_dfs.append(pred_score[['team','Total']])
    score_df = pd.concat(score_dfs)
    
    finalPicks = _finalizePickDataset(df, score_df, leagueId)
    _savePicks(finalPicks, year, saveLocation)
    return finalPicks

def getTeamScoreFromRosterConfig(df : pd.DataFrame, models : Dict[str, sklearn.pipeline.Pipeline]) -> pd.DataFrame:
    df_extended = df
    sum = 0
    score_dict = {}
    for pos in ['QB', 'RB', 'WR', 'TE', 'FLEX']:
        base_vars = ['A_pred_points',
            'A_std_error',
            'num_others',
            'avg_pred_points_other',
            'avg_std_error_other'
            ]
        model = models[pos]
        if pos in['RB','WR']:
            base_vars = base_vars + ['B_pred_points', 'B_std_error']

        base = model.predict(df_extended.loc[df_extended['FantPos'] == pos, base_vars])[0]
        score_dict[pos] = base
        sum += base
    score_dict['Total'] = sum
    return pd.DataFrame([score_dict])

if __name__ == '__main__':
    pd.options.display.max_columns = None
    
    path = pathlib.Path(__file__).parent.resolve()
    os.chdir(path)

    # ============
    # Create data for roster config regression
    # ============
    # st = time.time()
    # for i in range(2500):
    #     if i % 50 == 0:
    #         logger.info(f"Starting iteration {i}")
    #         logger.info(f"10 its took {time.time()  - st}")
    #         st = time.time()
    #     for year in range(2016, 2023):
    #         a = makeDataForRegression(year, 4, ScoringType.HPPR)
    #         path = f'../../data/regression/rosterConfig/rosterConfigData1.csv'
    #         if os.path.exists(path):
    #             a.to_csv(path, mode = 'a', header = False, index = False)
    #         else:
    #             a.to_csv(path, index = False)
    
    # ============
    # Create data for query
    # ============
    models : Dict[str, sklearn.pipeline.Pipeline] = initializeModels()    
    st = time.time()
    for i in range(7000):
        if i % 50 == 0:
            logger.info(f"Starting iteration {i}")
            logger.info(f"50 its took {time.time()  - st}")
            st = time.time()
        for year in range(2023, 2024):
            makeDataForQuery(year, 4, ScoringType.HPPR, models = models, saveLocation = '../../data/regression/queryableDraftPicks_2_{}.csv')
    
