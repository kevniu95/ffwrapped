from enum import Enum

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