import pathlib
import os
from typing import Dict, Any, List
import datetime as datetime
import time
import requests
import pickle
import glob
from io import StringIO
from websockets import client
import random

import pandas as pd
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod
import asyncio
from pyppeteer import launch
from aiolimiter import AsyncLimiter

from ..domain.common import ScoringType, thisFootballYear, WEEKLY_STATS_PARAMS
from ..util.logger_config import setup_logger
from ..util.config import Config

logger = setup_logger(__name__)

# Dictionary mapping fantasydata.com team abbreviations to pro-football-reference team abbreviations
ADP_TO_PFR = {'ARI':'ARI','ATL':'ATL','BAL':'BAL','BUF':'BUF','CAR':'CAR',
                'CHI':'CHI','CIN':'CIN','CLE':'CLE','DAL':'DAL','DEN':'DEN',
                'DET':'DET','GB':'GNB','HOU':'HOU','IND':'IND','JAX':'JAX',
                'KC':'KAN','LAC':'LAC','LAR':'LAR','LV':'LVR','MIA':'MIA',
                'MIN':'MIN','NO':'NOR','NE':'NWE','NYG':'NYG','NYJ':'NYJ',
                'PHI':'PHI','PIT':'PIT','SEA':'SEA','SF':'SFO','TB':'TAM',
                'TEN':'TEN','WAS':'WAS'}
PFR_LINK = 'https://www.pro-football-reference.com/years/{yr}/fantasy.htm'
WEEKLY_STATS_OFFSET_INTERVAL = 200


class Importer(ABC):
    def __init__(self, fullSavePath : str):
        self.fullSavePath = fullSavePath

    @abstractmethod
    def doImport(self):
        pass
    
    def _save(self, object : Any, savePath : str = None, **kwargs) -> None:
        if not savePath:
            savePath = self.fullSavePath
        if isinstance(object, pd.DataFrame) and savePath.endswith('.csv'):
            object.to_csv(savePath, **kwargs)
        else:
            with open(savePath, 'wb') as handle:
                pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"Successfully saved {self.__class__.__name__} object to {savePath}")
    
class PointsImport(Importer):
    def __init__(self, 
                 fullSavePath : str, 
                 year = thisFootballYear()):
        super().__init__(fullSavePath)
        self.year = year

    def doImport(self, 
                 link : str = PFR_LINK, 
                 save : bool = False, 
                 start_year :int = 2013, 
                 end_year: int = None) -> Dict[int, pd.DataFrame]:
        if not end_year:
            end_year = self.year
        logger.info(f"Importing points data from {start_year} to {end_year}...")
        
        fpts_dict = {}
        for i in range(start_year, end_year + 1):
            print(i)
            use_link = link.format(yr = i)
            page = requests.get(use_link)
            if page.status_code == 200:
                soup = BeautifulSoup(page.content, 'html.parser')
                
            table = soup.find_all('table', id = 'fantasy')
            df = pd.read_html(str(table), flavor = 'html5lib')[0]
            df.columns = df.columns.get_level_values(1)
            df['Year'] = i
            df.loc[df['FantPos'] == 'FB', 'FantPos'] = 'RB' # Change FB to RB
            df = df[df['Rk'] != 'Rk'].reset_index().drop('index', axis = 1)
            
            # Add pfref identifier
            data_append_csv_list = [row.td.get('data-append-csv') for row in table[0].find_all('tr') if row.td and row.td.get('data-append-csv')]
            df['pfref_id'] = data_append_csv_list
            
            # Add to Dict
            fpts_dict[i] = df
        if save:
            self._save(fpts_dict)
        return fpts_dict

class RosterImport(Importer):
    def __init__(self, 
                 fullSavePath : str, 
                 year = thisFootballYear()):
        self.fullSavePath = fullSavePath
        self.csvSavePath = fullSavePath.replace('.p', '{}.csv')
        self.csvSavePath = self.csvSavePath.replace('/created', '/created/roster_csv')
        super().__init__(self.fullSavePath)
        self.year = year
    
    async def _scrape_table_to_df(self, url: str, limiter: AsyncLimiter) -> pd.DataFrame:
        async with limiter:
            browser = await launch()
            page = await browser.newPage()
            await page.goto(url, {'timeout' : 10000})
            await page.waitForSelector('table#roster')
            
            # Select all td elements with a data-append-csv attribute
            tds = await page.querySelectorAll('td[data-append-csv]')

            # Extract the data-append-csv values and player names
            player_data = {}
            for td in tds:
                data_append_csv = await page.evaluate('(element) => element.getAttribute("data-append-csv")', td)
                player_name = await page.evaluate('(element) => element.querySelector("a").innerText', td)
                # strip player name of text in parentheses (e.g. (R))
                player_name = player_name.split('(')[0].strip()
                player_data[player_name] = data_append_csv
            
            # Extract entire html table
            table_html = await page.evaluate(
                '''() => document.querySelector('table#roster').outerHTML'''
            )
            await browser.close()

            df = pd.read_html(table_html)[0]
            df['tm'] = url.split('/')[4].lower()
            df = df[df['No.'] != 'No.'].reset_index().drop('index', axis = 1).copy()
            # strip player name of text in parentheses (e.g. (R))
            df['Player'] = df['Player'].str.split('(').str[0].str.strip()
            
            # Convert player_data dict to a DataFrame and merge with df
            player_df = pd.DataFrame(list(player_data.items()), columns=['Player', 'ID'])
            df = pd.merge(df, player_df, on='Player', how='left')
            return df
    
    async def _scrape_all_teams(self, teams: List[str], year: int) -> List[pd.DataFrame]:
        limiter = AsyncLimiter(20, 60)
        tasks = []
        for team in teams:
            # await asyncio.sleep(random.uniform(0.5, 2))
            endpoint = f'https://www.pro-football-reference.com/teams/{team.lower()}/{year}_roster.htm'        
            tasks.append(self._scrape_table_to_df(endpoint, limiter))
        return await asyncio.gather(*tasks)
    
    def _saveChunkToRosterCsv(self, chunk: List[str], year: int, save: bool) -> None:
        if not year:
            year = self.year
        rosters_2023 = asyncio.get_event_loop().run_until_complete(self._scrape_all_teams(chunk, year))
        rosters_df = pd.concat(rosters_2023)
        if save:
            mode = 'w'; header = True
            if os.path.exists(self.csvSavePath.format(str(year))):
                mode = 'a'
                header = False
            self._save(rosters_df, savePath = self.csvSavePath.format(str(year)), mode = mode, header = header)

    def doImportYear(self, year: int, save : bool = False, chunkSize: int = 18) -> pd.DataFrame:
        if not year:
            year = self.year
        logger.info(f"Scraping rosters for {year}...")
        
        pfr_abbrv_list = [i.lower() for i in list(pd.read_csv('../../data/imports/helpers/abbreviations.csv')['pfr'])]
        pfr_abbrv_list = [pfr_abbrv_list[i:i + chunkSize] for i in range(0, len(pfr_abbrv_list), chunkSize)]
        for chunk in pfr_abbrv_list:
            logger.info(f"Scraping chunk with {len(chunk)} teams...")
            self._saveChunkToRosterCsv(chunk, year, save)
            if chunk != pfr_abbrv_list[-1]:
                time.sleep(60)
        return pd.read_csv(self.csvSavePath.format(str(year)))

    def doImport(self, start_year: int = 2016, end_year: int = thisFootballYear(), save: bool = False) -> pd.DataFrame:
        logger.info(f"Scraping rosters from {start_year} to {end_year}...")
        for i in range(start_year, end_year + 1):
            self.doImportYear(i, save)
        
        # Import all roster{year}.csv files and concatenate into one DataFrame
        csv_folder = self.csvSavePath.replace('{}.csv', '')
        csv_files = glob.glob(f'{csv_folder}*.csv')
        df_list = []
        for f in csv_files:
            df = pd.read_csv(f)
            df['Year'] = int(f.split('/')[-1].split('.')[0][-4:])
            df_list.append(df)
        final_df = pd.concat(df_list)
        
        if save:
            self._save(final_df)
        return final_df
        
class ADPImport(Importer):
    def __init__(self, 
                 fullSavePath: str, 
                 year = thisFootballYear(), 
                 scoring_type: ScoringType = ScoringType.PPR):
        super().__init__(fullSavePath)
        self.year = year
        self.scoring_type = scoring_type
        
    def doImport(self, 
                 files_loc: str = '../../data/imports/helpers/historical_adp', 
                 save: bool = False) -> pd.DataFrame:
        logger.info(f"Importing ADP data through {self.year}...")
        df_dict = self._import_adp_data(files_loc = files_loc)
        df = self._prep_adp_df(df_dict)

        if self._checkColNames(df_dict) and self._checkTeamNames(df):
            if save: 
                self._save(df)
            return df
            
    def _import_adp_data(self, files_loc: str) -> Dict[str, pd.DataFrame]:
        # Source: https://fantasydata.com/nfl/adp - PPR
        df_dict = {}

        for i in range(2014, self.year + 1):
            cols = ["Name", "Team", "Position", "PositionRank", self.scoring_type.adp_column_name()]
            tmp= pd.read_csv(f'{files_loc}/{self.scoring_type.lower_name()}-adp-{i}.csv',
                            usecols = cols)
            tmp['Year'] = i
            df_dict[i] = tmp
        return df_dict

    def _prep_adp_df(self, adp_data : Dict[str, pd.DataFrame]) -> pd.DataFrame:
        # Limit to only top 200 in ADP per year
        # 1. Concat
        adp_df = pd.concat(adp_data.values())
        # 2. Re-order columns
        adp_df = adp_df[['Name', 'Year', 'Team', 'Position', 'PositionRank', self.scoring_type.adp_column_name()]]

        # 3. Get position rank as a number
        adp_df['PositionRank'] = adp_df['PositionRank'].str.extract('(\d+)')[0]
        
        # 4. Reset index
        adp_df.reset_index(inplace=True)
        adp_df.drop('index', axis = 1, inplace=True)
        # adp_df = adp_df.join(pd.get_dummies(adp_df['Position']))

        # 5. Remove III's from end of names
        adp_df['Name'] = adp_df['Name'].str.replace('([I ]+$)', '',regex= True)
        adp_df['Name'] = adp_df['Name'].str.replace('CJ ', 'C.J. ')
        adp_df['Name'] = adp_df['Name'].str.replace('DJ ', 'D.J. ')
        adp_df['Name'] = adp_df['Name'].str.replace('DK ', 'D.K. ')
        adp_df['Name'] = adp_df['Name'].str.replace('Steve Smith', 'Steve Smith Sr.')
        adp_df['Name'] = adp_df['Name'].str.replace('Marvin Jones Jr.', 'Marvin Jones', regex = False)
        adp_df['Name'] = adp_df['Name'].str.replace('Darrell Henderson Jr.', 'Darrell Henderson', regex = False)
        adp_df['Name'] = adp_df['Name'].str.replace('Gabe Davis', 'Gabriel Davis')
        # adp_df = adp_df[adp_df['AverageDraftPositionPPR'] < 173].copy()
        # Changing to 170 to have consistent cutoff for position-based regression

        # 6. Limit to standard, relevant fantas positions
        adp_df = adp_df[adp_df['Position'].isin(['RB','WR','QB','WR','TE'])]

        # 7. Update team names for those teams that have moved in last 10 years
        adp_df['Team'] = adp_df['Team'].replace(ADP_TO_PFR) 
        adp_df.loc[(adp_df['Team'] == 'LVR') & (adp_df['Year'] <= 2019),'Team'] = 'OAK'
        adp_df.loc[(adp_df['Team'] == 'LAC') & (adp_df['Year'] <= 2016),'Team'] = 'SDG'
        adp_df.loc[(adp_df['Team'] == 'LAR') & (adp_df['Year'] <= 2015),'Team'] = 'STL'

        adp_df.loc[adp_df['Name'].str.contains('Jordan Matthews'), 'Position'] = "WR"
        adp_df.loc[adp_df['Name'].str.contains('Funchess'), 'Position'] = "WR"
        adp_df.loc[adp_df['Name'].str.contains('Trubisky'), 'Name'] = "Mitchell Trubisky"
        adp_df.loc[adp_df['Name'].str.contains('Minshew'), 'Name'] = 'Gardner Minshew II'
        adp_df.loc[adp_df['Name'].str.contains('Chark'), 'Name'] = 'DJ Chark'
        adp_df.loc[adp_df['Name'].str.contains('Robert Griffin'), 'Name'] = 'Robert Griffin III'
        adp_df.loc[adp_df['Name'].str.contains('Willie Snead'), 'Name'] = 'Willie Snead'
        adp_df.loc[adp_df['Name'].str.contains('William Fuller'), 'Name'] = 'Will Fuller'
        adp_df.loc[adp_df['Name'].str.contains('Ronald Jones'), 'Name'] = 'Ronald Jones II'
        adp_df.loc[adp_df['Name'].str.contains('Benjamin Watson'), 'Name'] = 'Ben Watson'
        adp_df.loc[adp_df['Name'].str.contains('Rob Kelley'), 'Name'] = 'Robert Kelley'
        adp_df.loc[adp_df['Name'].str.contains('Henry Ruggs'), 'Name'] = 'Henry Ruggs III'
        adp_df.loc[adp_df['Name'].str.contains('Kenneth Walker'), 'Name'] = 'Kenneth Walker III'
        return adp_df 
        
    def _checkColNames(self, df_dict : Dict[str, pd.DataFrame]) -> bool:
        a = None
        for k, v in df_dict.items():
            if a is None:
                a = list(v.columns)
            if list(v.columns) != a:
                print('Column name check failed... There are some years where df columns have different names')
                return False
        return True

    # Check Teams are correct
    def _checkTeamNames(self, df : pd.DataFrame) -> bool:
        test = df[['Team','Year']].drop_duplicates().sort_values(['Year','Team'])
        a = test.groupby(['Team'], as_index = False).min()[['Team','Year']]
        b = test.groupby(['Team'], as_index = False).max()[['Team','Year']]
        c = a.merge(b, on = 'Team')
        c_sub = c[(c['Year_y'] - c['Year_x']) < 8]
        nameChanges = set(c_sub['Team'])
        changedNames = set(['LAC','LAR','LVR','OAK','SDG', 'STL'])
        if len(nameChanges.difference(changedNames)) > 0:
            print("Team name check failed...")
            print(c_sub[c_sub['Team'].isin(nameChanges.difference(changedNames))])
            return False
        return True
    
class WeeklyStatsImport(Importer):
    def __init__(self,
                 login: Dict[str, str],
                 fullSavePath: str = '../../data/imports/created/weekly_points_{}.csv'):
        super().__init__(fullSavePath)
        self.login = login
        self.max_offset_weeklyplayerstats = 6000
        self.weeklyStatsParams = WEEKLY_STATS_PARAMS
        self.base_url = 'https://stathead.com'
        self.browser = None
        
    async def _login_to_site(self, page: Any):
        login_path = '/users/login.cgi'
        await page.goto(f'{self.base_url}{login_path}', timeout = 10000)
        
        # Replace 'username' and 'password' with the appropriate name attributes for the login form
        await page.type('[name="username"]', self.login['username'])
        await page.type('[name="password"]', self.login['password'])

        # Simulate pressing Enter to submit the form
        await page.keyboard.press('Enter')
        
        # Wait for navigation to ensure the login process is complete
        await page.waitForNavigation({'timeout' : 10000})
    
    async def start(self):
        logger.debug("Launching browser...")
        self.browser = await launch(headless=True)  # headless=False to see the browser
        
    async def close(self):
        await self.browser.close()
    
    async def get_weekly_fantasy_stats_page(self, season: int, offset: int, page: Any) -> pd.DataFrame:
        logger.debug(f"Retrieving page {offset // 200 + 1}...")
        
        # Assuming self.weeklyStatsParams is already defined and includes all necessary parameters except 'offset', 'year_min', and 'year_max'
        self.weeklyStatsParams['offset'] = offset
        self.weeklyStatsParams['year_min'] = season
        self.weeklyStatsParams['year_max'] = season
        route = '/football/player-game-finder.cgi'
        query_string = '&'.join([f'{k}={v}' for k, v in self.weeklyStatsParams.items()])
        url_with_params = f'{self.base_url}{route}?{query_string}'
        
        logger.debug("Going to url_with_params...")
        await page.goto(url_with_params)
        content = await page.content()
        table = await self._get_single_pandas_table_from_content(content, element_id='stats')
        
        if table is not None:
            # Process the table as per your existing logic
            table['Rk'] = pd.to_numeric(table['Rk'], errors='coerce')
            table = table[table['Rk'].notnull()].reset_index(drop=True)
            
            # Example for extracting 'pfref_id', adapting to use BeautifulSoup for parsing
            soup = BeautifulSoup(content, 'html.parser')
            elems = soup.select('#div_stats td[data-append-csv]')
            new_field = [td['data-append-csv'] for td in elems]
            table['pfref_id'] = pd.Series(new_field)
        
        return table

    async def _get_single_pandas_table_from_content(self, content: str, element_id: str) -> pd.DataFrame:
        soup = BeautifulSoup(content, 'html.parser')
        table_html = soup.find('table', id=element_id)
        
        if table_html:
            table = pd.read_html(StringIO(str(table_html)), header=1)[0]
            return table
        else:
            print(f"Wasn't able to identify element with id {element_id}")
            return None    
        
    async def get_weekly_fantasy_stats_all(self, season: int, obs_limit: int = None, save: bool = False) -> pd.DataFrame:
        page = await self.browser.newPage()
        logger.debug("Logging in...")
        await self._login_to_site(page)
        
        offset_interval = 200
        tasks_per_time = 1
        data_dir = '../../data/imports/created/weekly_points/'
        data_file = f'{data_dir}weekly_points_{season}.p'

        if not obs_limit:
            obs_limit = self.max_offset_weeklyplayerstats

        offsets = list(range(0, obs_limit + 1, offset_interval))

        # Load the data dictionary if it exists, otherwise create an empty dictionary
        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                data_dict = pickle.load(f)
        else:
            data_dict = {}

        # Split offsets into chunks of tasks_per_minute
        offset_chunks = [offsets[i:i + tasks_per_time] for i in range(0, len(offsets), tasks_per_time)]

        for chunk in offset_chunks:
            tasks = {}
            for offset in chunk:
                # If the data for this offset already exists, skip it
                if offset in data_dict:
                    logger.info("Data for this offset already exists, skipping ...")
                    continue

                # Otherwise, create a task to get the data
                logger.info(f"Adding page {offset // 200 + 1} to set of tasks ...")
                tasks[offset] = self.get_weekly_fantasy_stats_page(season, offset, page)
                
            # If there are any tasks, run them concurrently
            if tasks:
                results = await asyncio.gather(*tasks.values())

                # Save the results in the data dictionary and save the dictionary
                for offset, result in zip(tasks.keys(), results):
                    if result is not None and not result.empty:
                        logger.info(f"Saving data for offset {offset} to pickle ...")
                        data_dict[offset] = result
                        with open(data_file, 'wb') as f:
                            pickle.dump(data_dict, f)
                
        # Combine all the data into a single DataFrame
        final_table = pd.concat(data_dict.values(), ignore_index=True)

        if save:
            self._save(final_table, savePath = self.fullSavePath.format(season))

        return final_table
    
    async def run(self, season: int = 2022, obs_limit: int = None, save : bool = True) -> pd.DataFrame:
        await self.start()
        final_table = await scraper.get_weekly_fantasy_stats_all(season=season, obs_limit=obs_limit, save=save)
        await scraper.close()
        return final_table  # This will return the DataFrame once the function completes
    
    def doImport(self, season: int, obs_limit: int = None, save: bool = True) -> pd.DataFrame:
        return asyncio.get_event_loop().run_until_complete(self.run(season, obs_limit, save))

        
if __name__ == '__main__':
    path = pathlib.Path(__file__).parent.resolve()
    os.chdir(path)
    print(os.getcwd())
    config = Config()
    
    # =======
    # Weekly Stats
    # =======
    stathead_login = {}
    stathead_login['username'] = config.parse_section('stathead')['username']
    stathead_login['password'] = config.parse_section('stathead')['password']
    
    scraper = WeeklyStatsImport(stathead_login, '../../data/imports/created/weekly_points/weekly_points_{}.csv')
    for i in range(2016, 2019):
        scraper.doImport(i)
    
    # =======
    # Roster
    # =======
    # roster_importer = RosterImport('../../data/imports/created/rosters.p')
    # df_roster = roster_importer.doImport(start_year = 2019, end_year = 2018, save = True)
    
    # =======
    # ADP
    # =======
    # adp_pickle = '../../data/imports/created/adp_full.p'
    # adp_pickle_nppr = '../../data/imports/created/adp_nppr_full.p'
    # adp_ppr_importer = ADPImport(adp_pickle)
    # df_adp_ppr = adp_ppr_importer.doImport(save = True)
    # adp_nppr_importer = ADPImport(adp_pickle_nppr, scoring_type = ScoringType.NPPR)
    # df_adp_nppr = adp_nppr_importer.doImport(save = True)

    # =======
    # Points
    # =======
    # points_pickle = '../../data/imports/created/points.p'
    # pt_importer = PointsImport(points_pickle)
    # fpts_dict = pt_importer.doImport(save = True)