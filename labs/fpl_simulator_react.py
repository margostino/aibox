import os
from langchain.agents import create_react_agent, load_tools, AgentExecutor
from langchain_openai import OpenAI
from langchain import hub
from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
# llm = OpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4")
#llm = OpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4-1106-preview")
llm = OpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo-instruct")

tools = load_tools(["graphql"],
                    graphql_endpoint="https://anfield-api-margostino.vercel.app/api/query",
                    llm=llm)
# prompt = hub.pull("hwchase17/openai-functions-agent")
prompt = hub.pull("hwchase17/react") # https://smith.langchain.com/hub/hwchase17/react?organizationId=8b40aa1a-1517-5571-a6d2-c117a8e9c4dd
agent = create_react_agent(llm, tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

query = """
INSTRUCTION: Using the following GraphQL query tell me which team scored (either teamAScore or teamHScore) the most goals in the match.
GRAPHQL QUERY:
{
	event(id: 1) {
		name
		fixtures {
			kickoffTime
			teamAName
			teamHName
			teamAScore
			teamHScore
		}
	}
}
Every element in "fixtures" object is a match. The field "teamAName" is the name of the team playing "away" while the field "teamHName" is the name of the team playing "home". 
The field "teamAScore" is the number of goals scored by the team playing "away" while the field "teamHScore" is the number of goals scored by the team playing "home". 
The field "kickoffTime" is the time when the match started.

Example:
```
Match 1:
Man City: 3
Burnley: 0

Match 2:
Nott'm Forest: 1
Arsenal: 2

Match 3:
West Ham: 1
Bournemouth: 1

Man City scored the most goals (3) in the match.
```
OUTPUT FORMAT:
Team: {team_name}
Goals: {goals}
"""
output_1 = agent_executor.invoke({"input": query})
print(output_1)

# output_2 = agent_executor.invoke({"input": "when you add 4  and 5 the result comes 10."})
# print(output_2)

