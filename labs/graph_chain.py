from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

graph = Neo4jGraph(
    url="bolt://localhost:7687", username="neo4j", password="password"    
)

# graph.refresh_schema()
# print(graph.schema)

chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0), graph=graph, verbose=True, top_k=100
)

chain.run("List of player of in Arsenal?")

