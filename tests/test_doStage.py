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

LAST_EXCLUDE_YEAR = 2015
ADJUSTED_PREDICT_YEAR = thisFootballYear()  - LAST_EXCLUDE_YEAR

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

@pytest.fixture(scope='module')
def finalDf(finalPtsDf : pd.DataFrame, finalAdpDf : pd.DataFrame) -> pd.DataFrame:
    return mergeAdpToPoints(finalPtsDf, finalAdpDf, SCORING)

def test_mergeAdpToPoints(finalDf: pd.DataFrame):
    # Everyone has these 7 fields
    assert finalDf[['Player','Tm','FantPos']].isnull().sum().sum() == 0
    assert finalDf[['QB','RB','TE','WR']].sum(axis=1).mean() == 1
    # 1. Check that only left-onlys don't have ADP info
    assert len(finalDf[finalDf[SCORING.adp_column_name()].isnull()]) == len(finalDf[finalDf['foundAdp'] == 'left_only'])
    # 2. Check that only right-onlys don't have pfref id or other info
    assert len(finalDf[finalDf['pfref_id'].isnull()]) == len(finalDf[finalDf['foundAdp'] == 'right_only'])
    
def test_AddFinalFields(finalDf: pd.DataFrame):
    finalDfNumCols = len(finalDf.columns)
    lastDf = addFinalFields(finalDf, LAST_EXCLUDE_YEAR)
    # 1. Check that AddFinalFields adds fields
    assert len(lastDf.columns) > finalDfNumCols
    # 2. Check that no rows are added or deleted
    assert len(lastDf) == len(finalDf)

def test_MakeFinalLimitations(finalDf: pd.DataFrame):
    base_vars_nonnull = ['Age', 
                        'adjYear', 
                        'drafted',
                        'rookie', 'Yrs',
                        'QB', 'RB', 'TE', 'WR']
    lastDf = makeFinalLimitations(finalDf, SCORING)

    # 1. Check all non-null fields are indeed non-null
    assert lastDf[base_vars_nonnull].isnull().sum().sum() == 0
    # 2. Check all players without PlayersAtPosition or PrvYrTmPtsAtPosition are on multiple teams
    teamSet = set(lastDf.loc[(lastDf['PlayersAtPosition'].isnull()) | (lastDf['PrvYrTmPtsAtPosition'].isnull()), 'Tm'].unique())
    for teamName in teamSet:
        assert teamName.endswith('TM')
    # 3. Check all players without AdpColumnName are leftOnly from foundAdp merge 
    assert lastDf[SCORING.adp_column_name()].isnull().sum() == (lastDf['foundAdp'] == 'left_only').sum()    
    # 4. Check less players than before
    assert len(lastDf) <= len(finalDf)