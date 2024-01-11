from ..util.webClient import RequestLimiter, GenericWebClient, SeleniumClient
from selenium.webdriver.remote.webelement import WebElement
from ..util.config import Config
import os
from bs4 import BeautifulSoup
import pandas as pd
from typing import Dict, List
import numpy as np
from abc import ABC, abstractmethod
import pathlib

from selenium.webdriver.common.by import By

BASE = 'https://www.pro-football-reference.com'

class DataValidationError(Exception):
    ''' Raised when data validation yields unexpected results '''
    pass

class NonUniqueTableError(Exception):
    def __init__(self, number : int):
        MESSAGE = 'Too many tables with given id: {number}. Expected just one.'
        super().__init__()
    pass

class WebScraper(ABC):
    def __init__(self, web_client : GenericWebClient, base : str = None):
        self.web_client : GenericWebClient = web_client
        self.base : str = base
        if not self.base:
            self.base = web_client.base
        
    def getBase(self) -> str:
        return self.base
    
    @abstractmethod
    def _getSinglePandasTableFromLink(self, endpoint : str, **kwargs):
        pass

class SeleniumWebScraper(WebScraper):
    def __init__(self, web_client : SeleniumClient, base : str = None):
        super().__init__(web_client, base)
    
    def _getSinglePandasTableFromLink(self, endpoint : str, **kwargs) -> pd.DataFrame:
        elementName = kwargs.get('elementName', None)
        self.web_client.get(endpoint)
        try:
            html_table : str = self.web_client.getOuterHtmlOfElementByXpath(elementName)
        except:
            print(f"Wasn't able to identify element {elementName} at current web page: {endpoint}")
            return None
        res = pd.read_html(html_table, header = 1, flavor = 'html5lib')
        
        if len(res) > 1:
            print("Identified more than one table - returning first one")    
        return res[0]
    
    def _getSpecificElementsFromCurrent(self, xpath_expression : str) -> List[WebElement]:
        return self.web_client.driver.find_elements(By.XPATH, xpath_expression)
        
        
class BeautifulSoupWebScraper(WebScraper):
    def __init__(self, web_client : GenericWebClient, base : str = None):
        super().__init__(web_client, base)
        
    def _getSinglePandasTableFromLink(self, endpoint : str, **kwargs) -> pd.DataFrame:
        table_specs : Dict[str, str] = kwargs.get('table_specs', None)
        params : Dict[str, str] = kwargs.get('params', None)

        resp = self.web_client.get(self.base + endpoint)
        soup = BeautifulSoup(resp.text, 'html.parser')
        html_tables = soup.find_all('table', table_specs)
        if len(html_tables) != 1:
            raise NonUniqueTableError(len(html_tables))
        return pd.read_html(str(html_tables[0]), flavor = 'html5lib')[0]

class RefScraper(BeautifulSoupWebScraper):
    def __init__(self, web_client : GenericWebClient, base : str = None):
        super().__init__(web_client, base)

    def getFantasyPlayers(self, year : int) -> pd.DataFrame():
        pass

    def getScheduleDfByYear(self, year : int, regular_only : bool = True) -> pd.DataFrame:
        tab = self._getSinglePandasTableFromLink(f'/years/{year}/games.htm', table_specs = {'id' : 'games'})
        
        tab = tab[tab['Week'].notnull()]
        exclude_weeks = ['Week']
        if regular_only:
            exclude_weeks += ['WildCard', 'Division', 'ConfChamp', 'SuperBowl']
        tab = tab[~tab['Week'].isin(exclude_weeks)]
        tab['datetime'] = pd.to_datetime(tab['Date'].astype(str) + ' ' + tab['Time'].astype(str), format="%Y-%m-%d %I:%M%p")
        
        tab['home'] = np.where(tab['Unnamed: 5'].isnull(), tab['Winner/tie'], tab['Loser/tie'])
        tab['away'] = np.where(tab['Unnamed: 5'].isnull(), tab['Loser/tie'], tab['Winner/tie'])
        return tab[['Week', 'home', 'away', 'datetime']]
        
    def getStandingsDfByYear(self, year : int, conf : str = None) -> pd.DataFrame:
        if conf:
            return self._getStandingsDfByYearByConf(year, conf)
        afc = self._getStandingsDfByYearByConf(year, 'AFC')
        nfc = self._getStandingsDfByYearByConf(year, 'NFC')
        keep_cols = ['Tm', 'W', 'L', 'T', 'PF', 'PA']
        return pd.concat((afc, nfc), axis = 0).reset_index().drop('index', axis = 1)[keep_cols]
    
    def _getStandingsDfByYearByConf(self, year : int, conf : str) -> pd.DateOffset:
        # Get html table
        tab = self._getSinglePandasTableFromLink(f'/years/{year}', table_specs = {'class' : 'sortable stats_table', 'id' : conf})
        tab = tab[~((tab['Tm'].str.startswith('AFC')) | (tab['Tm'].str.startswith('NFC')))]
        tab['Tm'] = tab['Tm'].str.replace(r'[^a-zA-Z]*$', '', regex=True)
        if tab.shape[0]!= 16:
            raise DataValidationError(f"Too many rows in dataset: {len(tab)}. Expected 16")
        return tab

class StatHeadScraper(SeleniumWebScraper):
    def __init__(self, web_client : SeleniumClient, login : Dict[str, str], base : str = None):
        super().__init__(web_client, base)
        self.login = login
        self._loginToSite()
        self.max_offset_weeklyplayerstats = 6000
        self.weeklyStatsParams = {'request' : '1',
                                    'draft_slot_min' : '1',
                                    'player_game_num_career_max' : '400',
                                    'comp_type' : 'reg',
                                    'order_by' : 'fantasy_points',
                                    'draft_year_max' : '2022',
                                    'season_start' : '1',
                                    'draft_pick_in_round' : 'pick_overall',
                                    'team_game_num_season_max' : '17',
                                    'team_game_num_season_min' : '1',
                                    'weight_max' : '500',
                                    'week_num_season_max' : '22',
                                    'rookie' : 'N',
                                    'conference' : 'any',
                                    'player_game_num_season_max' : '18',
                                    'year_min' : '2021',
                                    'qb_start_num_career_min' : '1',
                                    'draft_slot_max' : '500',
                                    'match' : 'player_game',
                                    'year_max' : '2021',
                                    'player_game_num_season_min' : '1',
                                    'draft_type' : 'R',
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

    def _loginToSite(self) -> None:
        login_path = '/users/login.cgi'
        self.web_client.get(self.base + login_path)
        self.web_client.login(self.login['username'], self.login['password'])
        
    def close_client(self) -> None:
        self.web_client.close()
    
    def getWeeklyFantasyStatsPage(self, season : int, offset : int) -> pd.DataFrame:
        print(f"Retrieving page {offset // 200 + 1}...")
        self.weeklyStatsParams['offset'] = offset
        self.weeklyStatsParams['year_min'] = season
        self.weeklyStatsParams['year_max'] = season
        endpoint = self.base + '/football/player-game-finder.cgi'
        query_string = '&'.join([f'{k}={v}' for k, v in self.weeklyStatsParams.items()])
        url_with_params = f'{endpoint}?{query_string}'
        tab = self._getSinglePandasTableFromLink(url_with_params, elementName = '//table[@id="stats"]')

        # Drop null fields
        tab['Rk'] = pd.to_numeric(tab['Rk'], errors = 'coerce')
        tab = tab[tab['Rk'].notnull()].reset_index().drop('index', axis = 1).copy()
        
        # Get ref ids
        elems = self._getSpecificElementsFromCurrent(xpath_expression = '//*[@id="div_stats"]//td[@data-append-csv]')
        new_field = [td.get_attribute('data-append-csv') for td in elems]
        tab['pfref_id'] = pd.Series(new_field) 
        return tab
    
    def getWeeklyFantasyStatsAll(self, season : int, obs_limit : int = None, save = False) -> pd.DataFrame:
        offset_interval : int = 200
        offset_current : int = 0
        all_tables : List[pd.DataFrame] = []
        done = False
        if not obs_limit:
            obs_limit = self.max_offset_weeklyplayerstats
        while not done and (offset_interval + offset_current <= obs_limit):
            current_table : pd.DataFrame = self.getWeeklyFantasyStatsPage(season, offset_current)
            if current_table is not None:
                all_tables.append(current_table)
                offset_current += offset_interval
            else:
                done = True
        final_table = pd.concat(all_tables)
        if save:
            final_table.to_csv(f'../../data/scraping/footballstats-weekly-points-{season}.csv')
        return final_table
    
def test():
    
    print(os.getcwd())
    # pd.read_csv('../data/scraping/footballstats-weekly-points-2022.csv')

    # def saveWeeklyFantasyStats(self) -> None:
    #     df = self.getWeeklyFantasyStats()
    #     # Save df to csv
    #     pass

    # def getWeeklyFantasyStats(self, year : int) -> pd.DataFrame():
    #     self.weeklyStatsParams['year_min'] = year
    #     self.weeklyStatsParams['year_max'] = year
    #     return self.getWeeklyFantasyStatsPage(0)

    # def getWeeklyFantasyStatsPage(self, offset : int) -> pd.DataFrame():
    #     self.weeklyStatsParams['offset'] = offset
    #     endpoint = '/football/player-game-finder.cgi'
    #     table_specs = {'id' : 'stats'}
    #     tab = self._getSinglePandasTableFromLink(endpoint, table_specs, self.weeklyStatsParams)
    #     return tab

    # def _getSinglePandasTableFromLink(self, endpoint : str, **kwargs):
    #     pass


if __name__ == '__main__':
    path = pathlib.Path(__file__).parent.resolve()
    os.chdir(path)
    
    # config = Config()
    # ref_base_link = config.parse_section('reader')['ref_base']
    # stathead_base_link = config.parse_section('reader')['stathead_base']
    # config_scraper = config.parse_section('requestLimiter')
    # print(config_scraper)
    # print(os.getcwd())

    # webclient = RequestLimiter(ref_base_link, **config_scraper)
    # scraper = RefScraper(webclient)
    # # print(scraper.getStandingsDfByYear(2022))
    # # print(scraper.getScheduleDfByYear(2022))
    # # test()

    # stathead_login = {}
    # stathead_login['username'] = config.parse_section('stathead')['username']
    # stathead_login['password'] = config.parse_section('stathead')['password']
    # webclient1 = SeleniumClient(stathead_base_link)
    # scraper1 = StatHeadScraper(webclient1, stathead_login)
    # scraper1.getWeeklyFantasyStatsAll(2022, obs_limit = None, save = True)

    # for i in range(2021, 2022):
    #     scraper1.getWeeklyFantasyStatsAll(i, obs_limit = None, save = True)
    # scraper1.close_client()


    # print(scraper.getScheduleDfByYear(2022))
    # import requests
    # a = requests.get('https://stathead.com/football/player-game-finder.cgi')
    # print(a)
    # print(a.text)
    # print(scraper1.getWeeklyFantasyStatsPage(0))