import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pytest
from src.modules.doPrep import *
from src.util.logger_config import setup_logger
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

def test_points_performSteps():
    pass

def main():
    # test_roster_loadData()
    # test_roster_performSteps()
    test_points_loadData()

if __name__ == '__main__':
    main()

