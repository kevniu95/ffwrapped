from enum import Enum
import datetime

class ScoringType(Enum):
    PPR = 1
    HPPR = 0.5
    NPPR = 0

    def points_name(self) -> str:
        return 'Pts_' + self.name
    
    def lower_name(self) -> str:
        return self.name.lower()
    
    def adp_column_name(self) -> str:
        if self.name == 'PPR':
            return 'AverageDraftPositionPPR'
        elif self.name == 'NPPR':
            return 'AverageDraftPosition'
        elif self.name == 'HPPR':
            return 'AverageDraftPositionHPPR'
        else:
            return None

def thisFootballYear() -> int:
    # If current date is between August and December, use this year
    # Otherwise, round down year down
    year = datetime.datetime.now().year
    if 8 <= datetime.datetime.now().month <= 12:
        return year
    else:
        return year - 1