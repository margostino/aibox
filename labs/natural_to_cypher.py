import openai
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

# OpenAI API key (replace with your own)
openai.api_key = os.environ['OPENAI_API_KEY']

# Connect to Neo4j
uri = "bolt://localhost:7687"
username = "neo4j"
password = "password"
driver = GraphDatabase.driver(uri, auth=(username, password))

def run_cypher_query(query):
    with driver.session() as session:
        return session.run(query)

# Define a function to convert natural language to Cypher using OpenAI
def to_cypher(nl_query):
    prompt = f"Translate the following natural language query about Fantasy Premier League to a Cypher query: '{nl_query}'"
    response = openai.Completion.create(prompt=prompt, max_tokens=150)
    cypher_query = response.choices[0].text.strip()
    return cypher_query

# Main function to run Langchain
def run_query(nl_query):
    cypher_query = to_cypher(nl_query)
    results = run_cypher_query(cypher_query)
    return results

# Example usage
query = "Which players play for Aston Villa?"
results = run_query(query)
print(results)
