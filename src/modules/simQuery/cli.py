from typing import List
import logging
import pandas as pd

from ...util.logger_config import setup_logger
from .simulationQuery import SimulationQueryService
LOG_LEVEL = logging.INFO
logger = setup_logger(__name__, level = LOG_LEVEL)

class SimulationQueryCLI():
    def __init__(self,
                 selectedTeam: int = None):
        self.runner = SimulationQueryService()
        self.selectedTeam = selectedTeam

    def _selectTeam(self):
        logger.debug("In select team method of CLI...")
        selectedTeam = self.selectedTeam
        while not selectedTeam:
            selectedTeam = input("Which team (int value 1-10) would you like to draft for?\n")
            try:
                selectedTeam = int(selectedTeam)
            except:
                selectedTeam = None
            if not isinstance(selectedTeam, int) or selectedTeam < 1 or selectedTeam > 10:
                print("Invalid selection. Please enter a number between 1 and 10.")
                selectedTeam = None
        return selectedTeam

    def _doPreselectPrints(self, 
                            roundNumber: int,
                            pickNumber: int, 
                            availablePlayers: pd.DataFrame,
                            conditions: List[str],
                            expectedPoints: float):
        print(f"\n========== Round {roundNumber} Info ==========")
        print(f"Round: {roundNumber} \ Pick: {pickNumber- (roundNumber - 1) * 10} \ Overall: {pickNumber}")
        print(f"Previous selections {conditions}")
        print(f"Expected max points for your team going into this selection: {round(expectedPoints, 2)}")
        # format availablePlayers['count'].sum() / 2 to int with commas
        print(f"Available samples backing this draft path: {int(availablePlayers['count'].sum() / 2):,}")
        print("====================================\n")
        print(f"Ok team {self.selectedTeam}, please make your selection for Round {roundNumber}:")
        print(availablePlayers)

    def requestSelection(self, roundNumber: int, availablePlayers: pd.DataFrame, conditions: List[str]):
        # Query user input until valid seleciton is made
        enteredPlayerId = None
        enteredList = []
        goBack = False
        while not enteredPlayerId:
            enteredPlayerId = input("Who do you choose? Please enter pfref_id of your selection...\n")
            if enteredPlayerId == 'goback':
                if roundNumber == 1:
                    print("Cannot go back further. Please enter a valid selection...")
                    enteredPlayerId = None
                    continue
                roundNumber -= 1
                conditions.pop()
                print("Reversing last draft pick...")
                goBack = True 
                break
            enteredList = enteredPlayerId.split(' ')
            for entered in enteredList:
                if entered not in set(availablePlayers['pfref_id']):
                    print(f"Invalid selection: {entered}. Please enter pfref_id of your selection and try again...\n")
                    enteredPlayerId = None
                    break
        return roundNumber, enteredList, goBack
            
    def startup(self):
        logger.debug("In startup method of CLI...")
        print('================')
        print("Welcome to simquery!")
        print('================')

        print("We will be simulating a 10-team HPPR draft for the year 2023.\n")
        self.selectedTeam = self._selectTeam()
        roundNumber = 1
        conditions = []
        while roundNumber < 15:
            logger.info("Processing round number: {}".format(roundNumber))
            thisPickNumber, availablePlayers, expectedPoints = self.runner.getPreselectInfo(self.selectedTeam, roundNumber, conditions)
            self._doPreselectPrints(roundNumber, thisPickNumber, availablePlayers, conditions, expectedPoints)
            
            roundNumber, enteredList, goBack = self.requestSelection(roundNumber, availablePlayers, conditions)
            if goBack:
                print("I am going back now!")
                continue
            # Make selection
            selection = self.runner.makeSelection(enteredList, thisPickNumber)
            logger.debug(f"Appending to conditions something of type {type((thisPickNumber, selection))}")
            conditions.append((thisPickNumber, selection))
            logger.debug(f"Conditions: {conditions}")
            roundNumber += 1

if __name__ == "__main__":
    cli = SimulationQueryCLI()
    cli.startup()