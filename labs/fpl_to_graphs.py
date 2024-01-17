# import requests
# import pandas as pd

# # Fetch data from Fantasy Premier League API
# response = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/')
# data = response.json()
# players = data['elements']

# # Transform data
# df = pd.DataFrame(players)
# players_transformed = df[['web_name', 'team']].to_dict('records')

# # Simulate loading data into Neo4j and print relationships
# for player in players_transformed:
#     print(f"CREATE (p:Player {{name: '{player['web_name']}', team: {player['team']}}})")

from neo4j import GraphDatabase
import requests

# Database connection details
uri = "bolt://localhost:7687"  # Note: Port 7687 is for Bolt protocol
user = "neo4j"
password = "password"

# Initialize Neo4j connection
driver = GraphDatabase.driver(uri, auth=(user, password))

# Function to execute a Cypher query
def execute_query(driver, query):
    with driver.session() as session:
        session.run(query)

# Fetch data from Fantasy Premier League API
response = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/')
data = response.json()

# Extract player and team data
players = data['elements']
teams = data['teams']

def escape_apostrophes(s):
    return s.replace("'", "")

# Function to convert dictionary properties into Cypher SET string
def properties_to_cypher_string(properties):
    return ", ".join(f"{key}: '{value}'" for key, value in properties.items())

# MERGE and execute Cypher queries for Team nodes
for team in teams:
    team_name_escaped = escape_apostrophes(team['name'])
    properties = {k: v for k, v in team.items() if k not in ['id']}
    properties['name'] = team_name_escaped
    properties_str = properties_to_cypher_string(properties)
    team_query = f"MERGE (t:Team {{id: {team['code']}}}) SET t += {{{properties_str}}}"
    execute_query(driver, team_query)
    
# # MERGE and execute Cypher queries for Player nodes and Team-Player relationships
for player in players:
    player_web_name_escaped = escape_apostrophes(player['web_name'])
    player_first_name_escaped = escape_apostrophes(player['first_name'])
    player_second_name_escaped = escape_apostrophes(player['second_name']) 
    properties = {k: v for k, v in player.items() if k not in ['id']}
    properties['web_name'] = player_web_name_escaped
    properties['first_name'] = player_first_name_escaped
    properties['second_name'] = player_second_name_escaped
    properties_str = properties_to_cypher_string(properties)
    player_query = f"MERGE (p:Player {{name: '{player_web_name_escaped}', team_id: {player['team_code']}}}) SET p += {{{properties_str}}}"
    relationship_query = f"MATCH (p:Player {{team_id: {player['team_code']}}}), (t:Team {{id: {player['team_code']}}}) MERGE (p)-[:PLAYS_FOR]->(t)"
    execute_query(driver, player_query)
    execute_query(driver, relationship_query)

# Close the driver connection
driver.close()
