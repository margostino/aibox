from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI

graph = Neo4jGraph(
    url="bolt://localhost:7687", username="neo4j", password="password"    
)

graph.refresh_schema()
print(graph.schema)