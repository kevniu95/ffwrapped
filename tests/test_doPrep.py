import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pytest
from src.modules.doPrep import *

pd.options.display.max_columns = None
MODULE_DIR = "/Users/kniu91/Documents/kevins_folders/Projects/ffwrapped/src/modules"
os.chdir(MODULE_DIR)

def test_roster_loadData():
    # Check original roster data is distinct
    roster_source = '../../data/imports/created/rosters.p'
    rd = RosterDataset([roster_source])
    df = rd.loadData()
    assert(df.shape[0] == df.drop_duplicates().shape[0])
    

def test_roster_performSteps():
    roster_source = '../../data/imports/created/rosters.p'
    rd = RosterDataset([roster_source])
    og_df = rd.loadData()
    new_df = rd.performSteps()
    assert (og_df.shape[0] == new_df.shape[0])
    assert (og_df.shape[1] < new_df.shape[1])
    
def main():
    test_roster_loadData()
    # test_roster_performSteps()

if __name__ == '__main__':
    main()

