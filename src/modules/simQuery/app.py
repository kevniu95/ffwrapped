from flask import Flask, request, jsonify
import logging
from .simulationQuery import SimulationQueryRunner
from .cli import FIXED_SCORING_TYPE, FIXED_COLS, FIXED_YEAR
from ...util.logger_config import setup_logger

LOG_LEVEL = logging.DEBUG
logger = setup_logger(__name__, level = LOG_LEVEL)

app = Flask(__name__)
sqr = SimulationQueryRunner(FIXED_SCORING_TYPE, FIXED_COLS, FIXED_YEAR)

# Define a route for the GET request
@app.route('/drafts/summary', methods=['POST'])
def get_draft_summary():
    logger.info("Here we are, getting the draft summary...")
    
    # Extract data from the JSON body
    data = request.get_json()
    teamNumber = data.get('teamNumber')
    roundNumber = data.get('roundNumber')
    conditions = data.get('conditions')
    
    # Ensure that the required fields are provided
    if teamNumber is None or roundNumber is None or conditions is None:
        return jsonify({'error': 'Missing required parameters'}), 400
    try:
        teamNumber = int(teamNumber)
        roundNumber = int(roundNumber)
    except ValueError:
        return jsonify({'error': 'Invalid team or round number'}), 400
    
    # Assuming sqr.getPreselectInfo returns the expected data correctly
    conditions = [tuple(item) for item in conditions] # need to convert to tuple of tuples, JSON doesn't recognize tuple
    thisPickNumber, availablePlayers, expectedPoints = sqr.getPreselectInfo(teamNumber, roundNumber, conditions)
    availablePlayers_json = availablePlayers.to_json(orient='records')
    return jsonify({'pickNumber': thisPickNumber, 'availablePlayers': availablePlayers_json, 'expectedPoints': expectedPoints})

@app.route('/drafts/selection', methods=['POST'])
def make_draft_selection():
    logger.info("Here we are, making the draft selection...")

    # Extract data from the JSON body
    data = request.get_json()
    idList = data.get('idList')
    pickNumber = data.get('pickNumber')
    if idList is None or pickNumber is None:
        return jsonify({'error': 'Missing required parameters'}), 400
    try:
        pickNumber = int(pickNumber)
    except ValueError:
        return jsonify({'error': 'Invalid pick number'}), 400

    selectionString = sqr.makeSelection(idList, pickNumber)
    return jsonify({'selection': selectionString})
    
if __name__ == '__main__':
    # Run the Flask app on localhost at the default port 5000
    app.run(debug=True)