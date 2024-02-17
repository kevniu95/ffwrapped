import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pytest

from src.modules.makeData import *
from src.util.logger_config import setup_logger

LOG_LEVEL = logging.INFO
logger = setup_logger(__name__)

pd.options.display.max_columns = None
MODULE_DIR = "/Users/kniu91/Documents/kevins_folders/Projects/ffwrapped/src/modules"
SCORING = ScoringType.HPPR
DRAFT_TEMP = 4
os.chdir(MODULE_DIR)

@pytest.fixture(scope='module', params = [2016, 2017, 2018, 2019, 2020, 2021, 2022])
def finishedDraft(request) -> Draft:
    return simulateLeagueAndDraft(request.param, 4, SCORING)

@pytest.fixture(scope='module')
def baselinePointDict(finishedDraft: Draft) -> Dict[str, float]:
    df = finishedDraft.pool.df
    drafted = df.loc[df['team'].notnull()]
    undrafted = df.loc[df['team'].isnull()]
    points_name = SCORING.points_name()
    return getBaselinePointDict(undrafted, points_name)

@pytest.fixture(scope='module')
def baselineDf(finishedDraft: Draft, baselinePointDict: Dict[str, float]) -> pd.DataFrame:
    league = finishedDraft.league
    positions = ['QB', 'RB', 'WR', 'TE']
    return generateBaselines(baselinePointDict, league.teams, positions = positions)

def test_getBaselinePointDict(baselinePointDict : Dict[str, float], finishedDraft: Draft):
    df = finishedDraft.pool.df
    undrafted = df.loc[df['team'].isnull()]
    points_name = SCORING.points_name()
    
    undrafted = undrafted.sort_values(by=['FantPos', points_name], ascending=[True, False]).reset_index().drop('index', axis = 1)
    copyDict = {}
    currentSub = undrafted[undrafted['FantPos'] == 'QB'].copy().reset_index().drop('index', axis = 1)
    copyDict['QB'] = currentSub.loc[4, points_name]
    currentSub = undrafted[undrafted['FantPos'] == 'RB'].copy().reset_index().drop('index', axis = 1)
    copyDict['RB'] = currentSub.loc[9, points_name]
    currentSub = undrafted[undrafted['FantPos'] == 'WR'].copy().reset_index().drop('index', axis = 1)
    copyDict['WR'] = currentSub.loc[9, points_name]
    currentSub = undrafted[undrafted['FantPos'] == 'TE'].copy().reset_index().drop('index', axis = 1)
    copyDict['TE'] = currentSub.loc[4, points_name]

    # Check xth position of each position is what is expected
    assert copyDict == baselinePointDict

def test_generateBaselines(baselineDf: pd.DataFrame, baselinePointDict: Dict[str, float], finishedDraft: Draft ):
    league = finishedDraft.league
    
    # Validate number of rows and uniqueness
    assert len(baselineDf) == (len(baselinePointDict) * len(league.teams) * 18)
    assert baselineDf.drop_duplicates().shape[0] == len(baselineDf)
    
    # Validate that each position has one unique baseline value
    unique_ppr_per_pos = baselineDf.groupby('FantPos')['PPR'].nunique()
    assert all(unique_ppr_per_pos   == 1)

def test_generateY(baselineDf: pd.DataFrame, finishedDraft: Draft):
    df = finishedDraft.pool.df
    drafted = df.loc[df['team'].notnull()]
    y_df = generateY(drafted, baselineDf, SCORING, finishedDraft.year)
    assert 80 <= y_df['Pts_HPPR'].sum() / len(finishedDraft.league.teams) <= 115

def test_getRosterConfigVariables(finishedDraft: Draft):
    df = finishedDraft.pool.df
    drafted = df.loc[df['team'].notnull()].copy()
    league = finishedDraft.league

    rosterConfigVariables = getRosterConfigVariables(drafted, league)
    assert len(rosterConfigVariables) == len(league.teams) * len(['QB','RB','WR','TE','FLEX'])
    assert rosterConfigVariables.drop_duplicates(['team','FantPos']).shape[0] == rosterConfigVariables.shape[0]
