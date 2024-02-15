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
    
# # A
# https://stathead.com/football/player-game-finder.cgi?request=1&draft_slot_min=1&player_game_num_career_max=400&comp_type=reg&order_by=fantasy_points&draft_year_max=2022&season_start=1&draft_pick_in_round=pick_overall&team_game_num_season_max=17&team_game_num_season_min=1&weight_max=500&week_num_season_max=22&rookie=N&conference=any&player_game_num_season_max=18&year_min=2022&qb_start_num_career_min=1&draft_slot_max=500&match=player_game&year_max=2022&player_game_num_season_min=1&draft_type=R&season_end==1&qb_start_num_career_max=400&week_num_season_min=1&player_game_num_career_min=1&cstat%5B1%5D=targets&ccomp%5B1%5D=gt&cval%5B1%5D=0&cstat%5B2%5D=pass_att&ccomp%5B2%5D=gt&cval%5B2%5D=0&cstat%5B3%5D=rush_att&ccomp%5B3%5D=gt&cval%5B3%5D=0&cstat%5B4%5D=fgm&ccomp%5B4%5D=gt&cval%5B4%5D=0&offset=0

# # B
# https://stathead.com/football/player-game-finder.cgi?request=1&draft_pick_type=overall&player_game_num_career_max=400&comp_type=reg&order_by=fantasy_points&season_start=1&team_game_num_season_max=17&team_game_num_season_min=1&weight_max=500&week_num_season_max=22&rookie=N&timeframe=seasons&player_game_num_season_max=18&year_min=2022&qb_start_num_career_min=1&match=player_game&year_max=2022&player_game_num_season_min=1&season_end=-1&qb_start_num_career_max=400&week_num_season_min=1&player_game_num_career_min=1&cstat[1]=targets&ccomp[1]=gt&cval[1]=0&cstat[2]=pass_att&ccomp[2]=gt&cval[2]=0&cstat[3]=rush_att&ccomp[3]=gt&cval[3]=0&cstat[4]=fgm&ccomp[4]=gt&cval[4]=0&offset=200

# # C
# https://stathead.com/football/player-game-finder.cgi?request=1&draft_pick_type=overall&player_game_num_career_max=400&comp_type=reg&order_by=fantasy_points  &season_start=1&team_game_num_season_max=17&team_game_num_season_min=1&weight_max=500&week_num_season_max=22&rookie=N&timeframe=seasons&player_game_num_season_max=18&year_min=2022&qb_start_num_career_min=1&match=player_game&year_max=2022&player_game_num_season_min=1&season_end=-1&qb_start_num_career_max=400&week_num_season_min=1&player_game_num_career_min=1&cstat[1]=targets&ccomp[1]=gt&cval[1]=0&cstat[2]=pass_att&ccomp[2]=gt&cval[2]=0&cstat[3]=rush_att&ccomp[3]=gt&cval[3]=0&cstat[4]=fgm&ccomp[4]=gt&cval[4]=0&offset=200

# https://stathead.com/football/player-game-finder.cgi?request=1                        &player_game_num_career_max=400&comp_type=reg&order_by=fantasy_points  &season_start=1&team_game_num_season_max=17&team_game_num_season_min=1&weight_max=500&week_num_season_max=22&rookie=N                  &player_game_num_season_max=18&year_min=2022&qb_start_num_career_min=1&match=player_game&year_max=2022&player_game_num_season_min=1&season_end==1&qb_start_num_career_max=400&week_num_season_min=1&player_game_num_career_min=1&cstat[1]=targets&ccomp[1]=gt&cval[1]=0&cstat[2]=pass_att&ccomp[2]=gt&cval[2]=0&cstat[3]=rush_att&ccomp[3]=gt&cval[3]=0&cstat[4]=fgm&ccomp[4]=gt&cval[4]=0&offset=0

# &draft_type=R
# draft_slot_max=500&
# &conference=any&
# &draft_slot_min=1&draft_year_max=2022&draft_pick_in_round=pick_overall&
# # https://stathead.com/football/player-game-finder.cgi?request=1&draft_pick_type=overall&player_game_num_career_max=400&comp_type=reg&order_by=fantasy_points&season_start=1&team_game_num_season_max=17&team_game_num_season_min=1&weight_max=500&week_num_season_max=22&rookie=N&timeframe=seasons&player_game_num_season_max=18&year_min=2022&qb_start_num_career_min=1&match=player_game&year_max=2022&player_game_num_season_min=1&season_end=-1&qb_start_num_career_max=400&week_num_season_min=1&player_game_num_career_min=1&cstat[1]=targets&ccomp[1]=gt&cval[1]=0&cstat[2]=pass_att&ccomp[2]=gt&cval[2]=0&cstat[3]=rush_att&ccomp[3]=gt&cval[3]=0&cstat[4]=fgm&ccomp[4]=gt&cval[4]=0&offset=200
# # https://stathead.com/football/player-game-finder.cgi?request=1&draft_pick_type=overall&player_game_num_career_max=400&comp_type=reg&order_by=fantasy_points&season_start=1&team_game_num_season_max=17&team_game_num_season_min=1&weight_max=500&week_num_season_max=22&rookie=N&timeframe=seasons&player_game_num_season_max=18&year_min=2022&qb_start_num_career_min=1&match=player_game&year_max=2022&player_game_num_season_min=1&season_end=-1&qb_start_num_career_max=400&week_num_season_min=1&player_game_num_career_min=1&cstat[1]=targets&ccomp[1]=gt&cval[1]=0&cstat[2]=pass_att&ccomp[2]=gt&cval[2]=0&cstat[3]=rush_att&ccomp[3]=gt&cval[3]=0&cstat[4]=fgm&ccomp[4]=gt&cval[4]=0&offset=200
# #                                                    '?request=1&timeframe=seasons&year_min={}&year_max={}&ccomp%5B1%5D=gt&cval%5B1%5D=0&cstat%5B1%5D=targets&ccomp%5B2%5D=gt&cval%5B2%5D=0&cstat%5B2%5D=pass_att&ccomp%5B3%5D=gt&cval%5B3%5D=0&cstat%5B3%5D=rush_att&ccomp%5B4%5D=gt&cval%5B4%5D=0&cstat%5B4%5D=fgm'
WEEKLY_STATS_PARAMS = {'request' : '1',
                                    # 'draft_slot_min' : '1',
                                    'draft_pick_type' : 'overall',
                                    'player_game_num_career_max' : '400',
                                    'comp_type' : 'reg',
                                    'order_by' : 'fantasy_points',
                                    # 'draft_year_max' : '2022',
                                    'season_start' : '1',
                                    # 'draft_pick_in_round' : 'pick_overall',
                                    'team_game_num_season_max' : '17',
                                    'team_game_num_season_min' : '1',
                                    'weight_max' : '500',
                                    'week_num_season_max' : '22',
                                    'rookie' : 'N',
                                    'timeframe': 'seasons',
                                    # 'conference' : 'any',
                                    'player_game_num_season_max' : '18',
                                    'year_min' : '2021',
                                    'qb_start_num_career_min' : '1',
                                    # 'draft_slot_max' : '500',
                                    'match' : 'player_game',
                                    'year_max' : '2021',
                                    'player_game_num_season_min' : '1',
                                    # 'draft_type' : 'R',
                                    'season_end=' : '1',
                                    'qb_start_num_career_max' : '400',
                                    'week_num_season_min' : '1',
                                    'player_game_num_career_min' : '1',
                                    'cstat[1]' : 'targets',
                                    'ccomp[1]' : 'gt',
                                    'cval[1]' : '0',
                                    'cstat[2]' : 'pass_att',
                                    'ccomp[2]' : 'gt',
                                    'cval[2]' : '0',
                                    'cstat[3]' : 'rush_att',
                                    'ccomp[3]' : 'gt',
                                    'cval[3]' : '0',
                                    'cstat[4]' : 'fgm',
                                    'ccomp[4]' : 'gt',
                                    'cval[4]' : '0',
                                    'offset' : '0'}