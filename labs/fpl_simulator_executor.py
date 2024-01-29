from langchain.agents import AgentExecutor, create_openai_functions_agent, load_tools
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain import hub


load_dotenv()

# prompt2 = hub.pull("hwchase17/openai-functions-agent")
# print(prompt2.messages)
# prompt = PromptTemplate(
#     input_variables=["question", "answer"], template="Question: {question}\n{answer}"
# )

prompt = PromptTemplate.from_template(
    """
    You are a helpful assistant. Your job is to search the premier league football fixtures and calculate the partial standings for every gameweek.
    In total you have to calculate standing for {gameweeks} gameweeks. You should return only the standings after each gameweek.
    
    A Gameweek is a set of fixtures (i.e. matches) played by the teams in the Premier League.
    Rules are as follows:
    - 3 points for a win
    - 1 point for a draw
    - 0 points for a loss
    
    The following GraphQL Query returns the results for each fixture for each event (i.e. Gameweek) and you should use it once per Gameweeks replacing the argument called "gameweek_id" with the id of the gameweek.
    The id for each gameweek is sequential and starts from 1. 
    So if you want to get the fixtures for the first gameweek you should replace the argument "gameweek_id" with 1, if you want to get the fixtures for the second gameweek you should replace the argument "gameweek_id" with 2, and so on.
	
    GRAPHQL QUERY:
	------------    
    {graphql_query}
    ------------    
    Every fixture is a match and it has the fields: teamAName, teamHName, teamAScore, teamHScore and id.
	Where:
	- teamAName: is the name of the team playing "away"
	- teamHName: is the name of the team playing "home"
	- teamAScore: is the number of goals scored by the team playing "away"
	- teamHScore: is the number of goals scored by the team playing "home"
	- id: id of the match
    
    MessagesPlaceholder: {agent_scratchpad}
    """ 
)

graphql_query = """
{
		event(id: gameweek_id) {
			name
			fixtures {
				id
				teamAName
				teamAScore
				teamHName			
				teamHScore
			}
		}
	}
"""

#llm = ChatOpenAI(model="gpt-4-1106-preview")
llm = ChatOpenAI(model="gpt-4")
#llm = ChatOpenAI(model="gpt-3.5-turbo-instruct")
tools = load_tools(["graphql"],
                    graphql_endpoint="https://anfield-api-margostino.vercel.app/api/query",
                    llm=llm)

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=100, max_execution_time=300, return_intermediate_steps=True)
response = agent_executor.invoke({"graphql_query": graphql_query, "gameweeks": 21})
#print(response["intermediate_steps"])