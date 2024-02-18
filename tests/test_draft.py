import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pytest
import os

from src.modules.draft import *
from src.util.logger_config import setup_logger
LOG_LEVEL = logging.INFO
logger = setup_logger(__name__)

pd.options.display.max_columns = None
MODULE_DIR = "/Users/kniu91/Documents/kevins_folders/Projects/ffwrapped/src/modules"
SCORING = ScoringType.HPPR
DRAFT_TEMP = 4
NUM_TEAMS = 10
COL_SUBSET = ['Player','Tm','Age','FantPos','Year','pfref_id','pred','var','var2','var_pred']
os.chdir(MODULE_DIR)

@pytest.fixture(scope='module', params = [2016, 2017, 2018, 2019, 2020, 2021, 2022])
def finishedDraft(request) -> Draft:
    return simulateLeagueAndDraft(request.param, DRAFT_TEMP, SCORING, NUM_TEAMS)

@pytest.mark.parametrize("year", [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023])
def test_initPlayerPoolDfFromRegDataset(year: int):
    col_subset = COL_SUBSET  + [SCORING.points_name(), SCORING.adp_column_name()]
    a = initPlayerPoolDfFromRegDataset(year, SCORING, col_subset)
    assert (a.shape[0] == a.drop_duplicates().shape[0])
    assert a['pred'].isnull().sum() == 0
    assert all([i in a.columns for i in ['Flex','team','pick']])

@pytest.mark.parametrize("numTeams", [8, 10, 12])
def test_initTeamsDefault(numTeams: int):
    league = League(numTeams, 2022)
    league.initTeamsDefault()
    assert len(league.teams) == numTeams
    assert len(set([i.id for i in league.teams])) == numTeams

# ====== 
# TODO: Update draft logic to account for 8 and 12-team leagues too
# ====== 
def test_simulateLeagueAndDraft(finishedDraft: Draft):
    sim_df = finishedDraft.pool.df
    numTeams = finishedDraft.league.numTeams
    rosterSpotsDrafted =  sum(ROSTER_CONFIG_DEFAULT.values()) - 2
    for i in range(1, numTeams + 1):
        i_str = 'team_' + str(i)
        assert len(sim_df[sim_df['team'] == i_str]) == rosterSpotsDrafted
        assert len(sim_df[sim_df['team'] == i_str]['pick'].drop_duplicates()) == rosterSpotsDrafted

def test_getRosterConfigOneTeam(finishedDraft: Draft):
    # Need to actually get simulated data to test this
    drafted = finishedDraft.pool.df[finishedDraft.pool.df['team'].notnull()].copy()
    league = finishedDraft.league
    drafted.sort_values(['team', 'FantPos', 'pred'], ascending = [True, True, False], inplace = True)
    for team in drafted['team'].unique():
        team = league.getTeamFromId(team)
        rows = team.getRosterConfigOneTeam(drafted)
        df = pd.DataFrame(rows, columns = rows[0].keys())
        assert len(df) == len(['QB','RB','WR','TE','FLEX'])
        assert df['team'].nunique() == 1
        assert df.loc[df['FantPos'] == 'QB', 'A_pred_points'].item() > 0
        assert df.loc[df['FantPos'] == 'RB', 'B_pred_points'].item() > 0
        assert df.loc[df['FantPos'] == 'WR', 'B_pred_points'].item() > 0
        assert df.loc[df['FantPos'] == 'TE', 'A_pred_points'].item() > 0
        assert df.loc[df['FantPos'] == 'FLEX', 'A_pred_points'].item() > 0