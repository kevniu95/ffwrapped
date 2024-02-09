import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pytest
from src.modules.doStage import *
from src.util.logger_config import setup_logger

LOG_LEVEL = logging.INFO
logger = setup_logger(__name__)

pd.options.display.max_columns = None
MODULE_DIR = "/Users/kniu91/Documents/kevins_folders/Projects/ffwrapped/src/modules"
SCORING = ScoringType.HPPR
os.chdir(MODULE_DIR)

@pytest.fixture(scope='module')
def finalRosterDf() -> pd.DataFrame:
    roster_source = '../../data/imports/created/rosters.p'
    return RosterDataset([roster_source]).performSteps()

@pytest.fixture(scope='module')
def finalPtsDf(finalRosterDf : pd.DataFrame) -> pd.DataFrame:
    pc = PointsConverter(SCORING)
    points_sources = ['../../data/imports/created/points.p']
    return PointsDataset(points_sources, SCORING, pc, currentRosterDf = finalRosterDf).performSteps()

@pytest.fixture(scope='module')
def finalAdpDf() -> pd.DataFrame:
    adp_sources = ['../../data/imports/created/adp_full.p',
                   '../../data/imports/created/adp_nppr_full.p']
    return ADPDataset(SCORING, adp_sources).performSteps()

def test_mergeAdpToPoints(finalPtsDf : pd.DataFrame, finalAdpDf : pd.DataFrame):
    df = mergeAdpToPoints(finalPtsDf, finalAdpDf, SCORING)
    # assert df[['Player','Tm','FantPos']].isnull().sum().sum() == 0
    # assert df[['QB','RB','TE','WR']].sum(axis=1).mean() == 1
    # print(df[df[SCORING.adp_column_name()].isnull()])
    assert len(df[df[SCORING.adp_column_name()].isnull()]) == len(df[df['foundAdp'] == 'left_only'])
    

    pass
