from typing import Dict, Any
import pandas as pd
import datetime as datetime
import requests
import pickle
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod
import pathlib
import os
# from .prepData import ScoringType
# from ..util.webClient import *
# from ..scraper.scraper import *

# Dictionary mapping fantasydata.com team abbreviations to pro-football-reference team abbreviations
ADP_TO_PFR = {'ARI':'ARI','ATL':'ATL','BAL':'BAL','BUF':'BUF','CAR':'CAR',
                'CHI':'CHI','CIN':'CIN','CLE':'CLE','DAL':'DAL','DEN':'DEN',
                'DET':'DET','GB':'GNB','HOU':'HOU','IND':'IND','JAX':'JAX',
                'KC':'KAN','LAC':'LAC','LAR':'LAR','LV':'LVR','MIA':'MIA',
                'MIN':'MIN','NO':'NOR','NE':'NWE','NYG':'NYG','NYJ':'NYJ',
                'PHI':'PHI','PIT':'PIT','SEA':'SEA','SF':'SFO','TB':'TAM',
                'TEN':'TEN','WAS':'WAS'}
PFR_LINK = 'https://www.pro-football-reference.com/years/{yr}/fantasy.htm'

class Importer(ABC):
    def __init__(self, fullSavePath : str):
        self.fullSavePath = fullSavePath
        self.willSave = False

    @abstractmethod
    def doImport(self):
        pass
    
    def _save(self, object : Any, savePath : str = None) -> None:
        if not savePath:
            savePath = self.fullSavePath
        with open(savePath, 'wb') as handle:
            pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved {self.__class__.__name__} object to {savePath}")
    
class PointsImport(Importer):
    def __init__(self, fullSavePath : str, year = datetime.date.today().year):
        super().__init__(fullSavePath)
        self.year = year

    def doImport(self, link : str = PFR_LINK, save : bool = False, start_year :int = 2013, end_year: int = None):
        if not end_year:
            end_year = self.year
        fpts_dict = {}
        for i in range(start_year, end_year):
            print(i)
            use_link = link.format(yr = i)
            page = requests.get(use_link)
            if page.status_code == 200:
                soup = BeautifulSoup(page.content, 'html.parser')
                
            table = soup.find_all('table', id = 'fantasy')
            df = pd.read_html(str(table), flavor = 'html5lib')[0]
            df.columns = df.columns.get_level_values(1)
            df['Year'] = i
            df = df[df['Rk'] != 'Rk'].reset_index().drop('index', axis = 1)
            
            # Add pfref identifier
            data_append_csv_list = [row.td.get('data-append-csv') for row in table[0].find_all('tr') if row.td and row.td.get('data-append-csv')]
            df['pfref_id'] = data_append_csv_list
            
            # Add to Dict
            fpts_dict[i] = df
        if save:
            self._save(fpts_dict)
        return fpts_dict
        
class RosterImport(SeleniumWebScraper):
    def __init__(self, fullSavePath : str, webClient : SeleniumClient = SeleniumClient('https://www.pro-football-reference.com/'), base = 'https://www.pro-football-reference.com/', year = datetime.date.today().year):
        super().__init__(webClient, base)
        print(self.web_client)
        self.fullSavePath = fullSavePath
        self.year = year
        self.webClient = webClient
    
    def _getSinglePandasTableFromLink(self, endpoint : str, **kwargs) -> pd.DataFrame:
        elementName = kwargs.get('elementName', None)
        self.web_client.get(endpoint)
        try:
            html_table : str = self.web_client.getOuterHtmlOfElementByXpath(elementName)
        except:
            print(f"Wasn't able to identify element {elementName} at current web page: {endpoint}")
            return None
        
        # Add ids Series to DataFrame
        res = pd.read_html(html_table, flavor = 'html5lib')
        if len(res) > 1:
            print("Identified more than one table - returning first one")    
        df = res[0]
        df = df[df['No.'] != 'No.'].reset_index().drop('index', axis = 1).copy()
        
        elems = self._getSpecificElementsFromCurrent(xpath_expression = '//*[@id="roster"]//td[@data-append-csv]')
        new_dict = {elem.text : elem.get_attribute('data-append-csv') for elem in elems}
        new_df = pd.DataFrame(list(new_dict.items()), columns=['Player', 'ID'])
        
        df = df.merge(new_df, on ='Player', how = 'left')
        print(df)
        return df
    
    def doImport(self) -> pd.DataFrame:
        pfr_abbrv_list = [i.lower() for i in list(pd.read_csv('../../data/import/abbreviations.csv')['pfr'])]
        for team in pfr_abbrv_list[4:5]:
            endpoint = f'https://www.pro-football-reference.com/teams/{team.lower()}/2023_roster.htm'        
            tab = self._getSinglePandasTableFromLink(endpoint, elementName = '//table[@id="roster"]')
            tab['tm'] = team
            roster2023path = '../../data/scraping/rosters2023.csv'
            if os.path.exists(roster2023path):
                tab.to_csv(roster2023path, header = False, mode = 'a')
            else:
                tab.to_csv(roster2023path)
        return pd.read_csv(roster2023path)

# class ADPImport(Importer):
#     def __init__(self, fullSavePath : str, year = datetime.date.today().year, scoring_type : ScoringType = ScoringType.PPR):
#         super().__init__(fullSavePath)
#         self.year = year
#         self.scoring_type = scoring_type
        
#     def doImport(self, files_loc : str = '../../data/research/historical_adp', save : bool = False) -> pd.DataFrame:
#         self.willSave = save
#         df_dict = self._import_adp_data(files_loc = files_loc)
#         df = self._prep_adp_df(df_dict)

#         if self._checkColNames(df_dict) and self._checkTeamNames(df):
#             if save: 
#                 self._save(df)
#             return df
            
#     def _import_adp_data(self, files_loc : str) -> Dict[str, pd.DataFrame]:
#         # Source: https://fantasydata.com/nfl/adp - PPR
#         df_dict = {}
#         for i in range(2014, self.year + 1):
#             cols = ["Name", "Team", "Position", "PositionRank", self.scoring_type.adp_column_name()]
#             tmp= pd.read_csv(f'{files_loc}/{self.scoring_type.lower_name()}-adp-{i}.csv',
#                             usecols = cols)
#             tmp['Year'] = i
#             df_dict[i] = tmp
#         return df_dict

#     def _prep_adp_df(self, adp_data : Dict[str, pd.DataFrame]) -> pd.DataFrame:
#         # Limit to only top 200 in ADP per year

#         # 1. Concat
#         adp_df = pd.concat(adp_data.values())
#         # 2. Re-order columns
#         adp_df = adp_df[['Name', 'Year', 'Team', 'Position', 'PositionRank', self.scoring_type.adp_column_name()]]

#         # 3. Get position rank as a number
#         adp_df['PositionRank'] = adp_df['PositionRank'].str.extract('(\d+)')[0]
        
#         # 4. Reset index
#         adp_df.reset_index(inplace=True)
#         adp_df.drop('index', axis = 1, inplace=True)
#         # adp_df = adp_df.join(pd.get_dummies(adp_df['Position']))

#         # 5. Remove III's from end of names
#         adp_df['Name'] = adp_df['Name'].str.replace('([I ]+$)', '',regex= True)
#         adp_df['Name'] = adp_df['Name'].str.replace('CJ ', 'C.J. ')
#         adp_df['Name'] = adp_df['Name'].str.replace('DJ ', 'D.J. ')
#         adp_df['Name'] = adp_df['Name'].str.replace('DK ', 'D.K. ')
#         adp_df['Name'] = adp_df['Name'].str.replace('Steve Smith', 'Steve Smith Sr.')
#         adp_df['Name'] = adp_df['Name'].str.replace('Marvin Jones Jr.', 'Marvin Jones', regex = False)
#         adp_df['Name'] = adp_df['Name'].str.replace('Darrell Henderson Jr.', 'Darrell Henderson', regex = False)
#         adp_df['Name'] = adp_df['Name'].str.replace('Gabe Davis', 'Gabriel Davis')
#         # adp_df = adp_df[adp_df['AverageDraftPositionPPR'] < 173].copy()
#         # Changing to 170 to have consistent cutoff for position-based regression

#         # 6. Limit to standard, relevant fantas positions
#         adp_df = adp_df[adp_df['Position'].isin(['RB','WR','QB','WR','TE'])]

#         # 7. Update team names for those teams that have moved in last 10 years
#         adp_df['Team'] = adp_df['Team'].replace(ADP_TO_PFR) 
#         adp_df.loc[(adp_df['Team'] == 'LVR') & (adp_df['Year'] <= 2019),'Team'] = 'OAK'
#         adp_df.loc[(adp_df['Team'] == 'LAC') & (adp_df['Year'] <= 2016),'Team'] = 'SDG'
#         adp_df.loc[(adp_df['Team'] == 'LAR') & (adp_df['Year'] <= 2015),'Team'] = 'STL'

#         adp_df.loc[adp_df['Name'].str.contains('Jordan Matthews'), 'Position'] = "WR"
#         adp_df.loc[adp_df['Name'].str.contains('Funchess'), 'Position'] = "WR"
#         adp_df.loc[adp_df['Name'].str.contains('Trubisky'), 'Name'] = "Mitchell Trubisky"
#         adp_df.loc[adp_df['Name'].str.contains('Minshew'), 'Name'] = 'Gardner Minshew II'
#         adp_df.loc[adp_df['Name'].str.contains('Chark'), 'Name'] = 'DJ Chark'
#         adp_df.loc[adp_df['Name'].str.contains('Robert Griffin'), 'Name'] = 'Robert Griffin III'
#         adp_df.loc[adp_df['Name'].str.contains('Willie Snead'), 'Name'] = 'Willie Snead'
#         adp_df.loc[adp_df['Name'].str.contains('William Fuller'), 'Name'] = 'Will Fuller'
#         adp_df.loc[adp_df['Name'].str.contains('Ronald Jones'), 'Name'] = 'Ronald Jones II'
#         adp_df.loc[adp_df['Name'].str.contains('Benjamin Watson'), 'Name'] = 'Ben Watson'
#         adp_df.loc[adp_df['Name'].str.contains('Rob Kelley'), 'Name'] = 'Robert Kelley'
#         adp_df.loc[adp_df['Name'].str.contains('Henry Ruggs'), 'Name'] = 'Henry Ruggs III'
#         adp_df.loc[adp_df['Name'].str.contains('Kenneth Walker'), 'Name'] = 'Kenneth Walker III'
#         return adp_df 
        
#     def _checkColNames(self, df_dict : Dict[str, pd.DataFrame]) -> bool:
#         a = None
#         for k, v in df_dict.items():
#             if a is None:
#                 a = list(v.columns)
#             if list(v.columns) != a:
#                 print('Column name check failed... There are some years where df columns have different names')
#                 return False
#         return True

#     # Check Teams are correct
#     def _checkTeamNames(self, df : pd.DataFrame) -> bool:
#         test = df[['Team','Year']].drop_duplicates().sort_values(['Year','Team'])
#         a = test.groupby(['Team'], as_index = False).min()[['Team','Year']]
#         b = test.groupby(['Team'], as_index = False).max()[['Team','Year']]
#         c = a.merge(b, on = 'Team')
#         c_sub = c[(c['Year_y'] - c['Year_x']) < 8]
#         nameChanges = set(c_sub['Team'])
#         changedNames = set(['LAC','LAR','LVR','OAK','SDG', 'STL'])
#         if len(nameChanges.difference(changedNames)) > 0:
#             print("Team name check failed...")
#             print(c_sub[c_sub['Team'].isin(nameChanges.difference(changedNames))])
#             return False
#         return True

if __name__ == '__main__':
    path = pathlib.Path(__file__).parent.resolve()
    os.chdir(path)
    print('a')
    
    # =======
    # Roster
    # =======
    roster_importer = RosterImport(fullSavePath = roster_2023_pickle, year = 2023)
    df_roster = roster_importer.doImport()
    # roster_2023_pickle = '../../data/research/created/roster2023.p'
    # pd.read_pickle(roster_2023_pickle)
    
    # print(df_roster)

    # # =======
    # # ADP
    # # =======
    # adp_pickle = '../../data/research/created/adp_full.p'
    # adp_pickle_nppr = '../../data/research/created/adp_nppr_full.p'
    # pd.read_pickle(adp_pickle)
    # pd.read_pickle(adp_pickle_nppr)
    # # adp_ppr_importer = ADPImport(adp_pickle)
    # # df_adp_ppr = adp_ppr_importer.doImport(save = True)
    # # adp_nppr_importer = ADPImport(adp_pickle_nppr, scoring_type = ScoringType.NPPR)
    # # df_adp_nppr = adp_nppr_importer.doImport(save = True)

    # # =======
    # # Points
    # # =======
    # points_pickle = '../../data/research/created/points.p'
    # pd.read_pickle(points_pickle)
    # # pt_importer = PointsImport(points_pickle)
    # # fpts_dict = pt_importer.doImport(save = True)
    
    # # New import on 8.31
    # points_2000_2012_pickle = '../../data/research/created/points_2000_2012.p'
    # # pt_importer1 = PointsImport(fullSavePath = points_2000_2012_pickle)
    # # pt_importer1.doImport(save = True, start_year = 2000, end_year = 2013)