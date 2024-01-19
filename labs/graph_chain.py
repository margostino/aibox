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
    ChatOpenAI(temperature=0, model_name='gpt-4-1106-preview'), graph=graph, verbose=True, top_k=100
)

#chain.run("List of player of in Arsenal?")
chain.run("Come up with a squad of 11 players. Each from different team. The squad should include: 1 goalkeeper (element_type=1), 3 defenders (element_type=2), 5 midfielder (element_type=3) and 2 forwards (element_type=4). You should only return the names and price of the players.")

