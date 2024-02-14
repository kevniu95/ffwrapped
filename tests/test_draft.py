import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pytest

from src.modules.draft import *
from src.util.logger_config import setup_logger

LOG_LEVEL = logging.INFO
logger = setup_logger(__name__)

pd.options.display.max_columns = None
MODULE_DIR = "/Users/kniu91/Documents/kevins_folders/Projects/ffwrapped/src/modules"
SCORING = ScoringType.HPPR
DRAFT_TEMP = 4
os.chdir(MODULE_DIR)

@pytest.mark.parametrize("year", [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023])
def test_initPlayerPoolDfFromRegDataset(year: int):
    a = initPlayerPoolDfFromRegDataset(year, SCORING, use_compressed = False)
    assert (a.shape[0] == a.drop_duplicates().shape[0])
    assert a['pred'].isnull().sum() == 0
    assert all([i in a.columns for i in ['Flex','team','pick']])

@pytest.mark.parametrize("numTeams", [8, 10, 12])
def test_initTeamsDefault(numTeams: int):
    league = League(numTeams)
    league.initTeamsDefault()
    assert len(league.teams) == numTeams
    assert len(set([i.id for i in league.teams])) == numTeams

# ====== 
# TODO: Update draft logic to account for 8 and 12-team leagues too
# ====== 
@pytest.mark.parametrize("numTeams", [10])
def test_simulateLeagueAndDraft(numTeams: int):
    simulated = simulateLeagueAndDraft(2023, DRAFT_TEMP, SCORING, numTeams)
    sim_df = simulated.pool.df
    sim_df.to_csv('sim.csv', index = False)
    rosterSpotsDrafted =  sum(ROSTER_CONFIG_DEFAULT.values()) - 2
    for i in range(1, numTeams + 1):
        i_str = 'team_' + str(i)
        assert len(sim_df[sim_df['team'] == i_str]) == rosterSpotsDrafted
        assert len(sim_df[sim_df['team'] == i_str]['pick'].drop_duplicates()) == rosterSpotsDrafted

