# https://www.safaribooksonline.com/library/view/python-microservices-development/9781785881114/d51714de-4d68-483a-a971-41037a91d788.xhtml
from flask import Blueprint, jsonify

teams = Blueprint('teams', __name__)

_DEVS = ['Tarek', 'Bob']
_OPS = ['Bill']
_TEAMS = {1: _DEVS, 2: _OPS}

@teams.route('/teams')
def get_all():
    return jsonify(_TEAMS)

@teams.route('/teams/<int:team_id>')
def get_team(team_id):
    return jsonify(_TEAMS[team_id])



blueprint = manager.create_api_blueprint(Person, methods=['GET',
                                                          'POST'])
app.register_blueprint(blueprint)