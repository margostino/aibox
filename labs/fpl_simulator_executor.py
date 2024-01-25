from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent, load_tools
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")
# Choose the LLM that will drive the agent
llm = ChatOpenAI(model="gpt-3.5-turbo-1106")
tools = load_tools(["graphql"],
                    graphql_endpoint="https://anfield-api-margostino.vercel.app/api/query",
                    llm=llm)

# Construct the OpenAI Functions agent
agent = create_openai_functions_agent(llm, tools, prompt)
# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=50, max_execution_time=120)

query = """
INSTRUCTION: Which team scored the most goals in a single match? 
You only need to consider the first 5 Gameweeks and return by Gameweek: the team name, the number of goals scored and the fixture id.
In case there are more than 1 team with most goals, take into account the goal difference (i.e. goals scored - goals conceded).
This is for example Team A scored 3 goals and conceded 1 goal, so the goal difference is 2.

OUTPUT FORMAT:
Gameweek: _gameweek_id_
Team: _team_name__with_most_scored_goals_
Score: _most_scored_goals_
Fixture: _fixture_id_


The following GraphQL query will return the fixtures of a single gameweek.
You should use the query many times as Gameweeks (gameweek_id) are. 
Every event is a Gameweek. Every Gameweek has fixtures (i.e. matches). 
Every match has the fields: teamAName, teamHName, teamAScore, teamHScore and kickoffTime.
Where:
- teamAName: is the name of the team playing "away"
- teamHName: is the name of the team playing "home"
- teamAScore: is the number of goals scored by the team playing "away"
- teamHScore: is the number of goals scored by the team playing "home"
- kickoffTime: is the time when the match started

GRAPHQL QUERY (request):
{
	event(id: {gameweek_id}) {
		name
		fixtures {
			id
			kickoffTime
			teamAName
            teamAScore
			teamHName			
			teamHScore
		}
	}
}

"""

agent_executor.invoke({"input": query})