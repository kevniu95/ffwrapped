import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pytest
from src.modules.doPrep import *
from src.util.logger_config import setup_logger

LOG_LEVEL = logging.INFO
logger = setup_logger(__name__)

pd.options.display.max_columns = None
MODULE_DIR = "/Users/kniu91/Documents/kevins_folders/Projects/ffwrapped/src/modules"
SCORING = ScoringType.HPPR
os.chdir(MODULE_DIR)

@pytest.fixture(scope='module')
def rosterDataset() -> RosterDataset:
    roster_source = '../../data/imports/created/rosters.p'
    return RosterDataset([roster_source])

@pytest.fixture(scope='module')
def pointsDataset(rosterDataset: RosterDataset) -> PointsDataset:
    pc = PointsConverter(SCORING)
    points_sources = ['../../data/imports/created/points.p']
    rd_performed = rosterDataset.performSteps()
    return PointsDataset(points_sources, SCORING, pc, currentRosterDf= rd_performed)

@pytest.fixture(scope='module')
def adpDataset() -> ADPDataset:
    adp_sources = ['../../data/imports/created/adp_full.p',
                   '../../data/imports/created/adp_nppr_full.p']
    return ADPDataset(SCORING, adp_sources) 

# ========= 
# Roster
# ========= 
def test_roster_loadData(rosterDataset: RosterDataset):
    # Roster input data should not have any duplicates
    df = rosterDataset.loadData()
    assert(df.shape[0] == df.drop_duplicates().shape[0])

def test_roster_performSteps(rosterDataset: RosterDataset):
    # Roster output data adds information, not rows
    og_df = rosterDataset.loadData()
    new_df =rosterDataset.performSteps()
    assert (og_df.shape[0] == new_df.shape[0])
    assert (og_df.shape[1] < new_df.shape[1])

# ========= 
# Points
# ========= 
def test_points_loadData(pointsDataset: PointsDataset):
    og_df = pointsDataset.loadData()
    assert og_df.shape[0] == og_df.drop_duplicates(['pfref_id','Year','Tm']).shape[0]

    years = set(og_df['Year'].unique())
    for i in range(2013, 2024):
        assert i in years
    
def test_points_groupMultiTeamPlayers(pointsDataset: PointsDataset):
    og_df = pointsDataset.loadData()
    new_df = pointsDataset._groupMultiTeamPlayers(og_df)
    assert new_df.duplicated(['pfref_id','Year']).sum() == 0
    assert new_df.shape[0] <= og_df.shape[0]

def test_points_createPreviousYear(pointsDataset: PointsDataset):
    og_df = pointsDataset.loadData()
    og_df = pointsDataset._groupMultiTeamPlayers(og_df)
    og_df = pointsDataset._filterPredictYear(og_df)
    inputSize = og_df.shape
    new_df = pointsDataset._createPreviousYear(og_df)
    # Assert only this predicted year doesn't have fantasy points
    assert new_df.loc[new_df['Pts_HPPR'].isnull(), 'Year'].unique() == pointsDataset.currentYear
    # Assert that all guys without PrvPts are properly marked as missingLastYear = 1
    assert new_df.loc[new_df['PrvPts_HPPR'].isnull(), 'missingLastYear'].mean() == 1
    outputSize = new_df.shape

def test_points_addCurrentRosters(pointsDataset: PointsDataset):
    og_df = pointsDataset.loadData()
    og_df = pointsDataset._groupMultiTeamPlayers(og_df)
    og_df = pointsDataset._filterPredictYear(og_df)
    new_df = pointsDataset._createPreviousYear(og_df)
    new_df = pointsDataset._addCurrentRosters(new_df)
    assert new_df.drop_duplicates(['pfref_id','Year']).shape[0] == new_df.shape[0]  
    assert new_df['Player'].isnull().sum() == 0
    
def test_points_performSteps(pointsDataset: PointsDataset):
    og_df = pointsDataset.loadData()
    og_df = pointsDataset._groupMultiTeamPlayers(og_df)
    og_df = pointsDataset._filterPredictYear(og_df)
    og_df = pointsDataset._createPreviousYear(og_df)
    og_df = pointsDataset._addCurrentRosters(og_df)

    new_df = pointsDataset.performSteps()

    # After adding current rosters, number of rows shouldn't change with subsequent steps
    assert og_df.shape[0] == new_df.shape[0]
    assert og_df.shape[1] < new_df.shape[1]
    
# ========= 
# ADP
# ========= 
def test_adp_loadData(adpDataset: ADPDataset):
    og_df = adpDataset.loadData()
    assert og_df.drop_duplicates(['Name','Year','Team','Position']).shape[0] == og_df.shape[0]
    assert list(og_df) == ['Name','Year','Team','Position',SCORING.adp_column_name()]
    assert og_df[SCORING.adp_column_name()].isnull().sum() == 0


def test_adp_performSteps(adpDataset: ADPDataset):
    og_df = adpDataset.loadData()
    new_df = adpDataset.performSteps()
    assert new_df.shape[0] < og_df.shape[0]
    assert og_df[SCORING.adp_column_name()].isnull().sum() == 0
    
def main():
    # test_roster_loadData()
    # test_roster_performSteps()
    test_points_loadData()

if __name__ == '__main__':
    main()