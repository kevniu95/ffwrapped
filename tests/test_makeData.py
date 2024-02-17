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
def finishedDraft(request) -> pd.DataFrame:
    return simulateLeagueAndDraft(request.param, 4, SCORING)

def test_getBaselinePointDict(finishedDraft: pd.DataFrame):
    df = finishedDraft.pool.df
    undrafted = df.loc[df['team'].isnull()]
    points_name = SCORING.points_name()
    baselinePoints = getBaselinePointDict(undrafted, points_name)

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
    assert copyDict == baselinePoints

dict1 = {'QB': 214.98, 'RB': 72.6, 'WR': 127.4, 'TE': 97.7}
dict2 = {'QB': 241.22, 'RB': 111.6, 'WR': 129.70000000000002, 'TE': 118.6}
dict3 = {'QB': 224.94, 'RB': 95.10000000000001, 'WR': 122.5, 'TE': 119.72}
@pytest.mark.parametrize("entryDict", [dict1, dict2, dict3])
def test_generateBaselines(entryDict: Dict[str, float]):
    league = League(10, 2022)
    league.initTeamsDefault()
    positions = ['QB', 'RB', 'WR', 'TE']
    baseslineDf = generateBaselines(entryDict, league.teams, positions = positions)

    # Validate number of rows and uniqueness
    assert len(baseslineDf) == (len(entryDict) * len(league.teams) * 18)
    assert baseslineDf.drop_duplicates().shape[0] == len(baseslineDf)
    
    # Validate that each position has one unique baseline value
    unique_ppr_per_pos = baseslineDf.groupby('FantPos')['PPR'].nunique()
    assert all(unique_ppr_per_pos   == 1)

def test_generateY(finishedDraft: pd.DataFrame):
    df = finishedDraft.pool.df
    drafted = df.loc[df['team'].notnull()]
    undrafted = df.loc[df['team'].isnull()]
    points_name = SCORING.points_name()
    baselinePoints = getBaselinePointDict(undrafted, points_name)
    appendMe = generateBaselines(baselinePoints, finishedDraft.league.teams)
    y_df = generateY(drafted, appendMe, SCORING, finishedDraft.year)
    # print(y_df.head(10))
    # print(y_df['Pts_HPPR'])
    # print(y_df['Pts_HPPR'].sum())
    print(finishedDraft.year)
    print(y_df['Pts_HPPR'].sum() / len(finishedDraft.league.teams))