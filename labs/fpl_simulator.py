import os
from langchain.agents import create_react_agent, load_tools, AgentType, AgentExecutor
from langchain_openai import OpenAI
from langchain import hub
from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
llm = OpenAI(openai_api_key=OPENAI_API_KEY)

tools = load_tools(["wikipedia", "llm-math", "graphql"],
                    graphql_endpoint="https://anfield-api-margostino.vercel.app/api/query",
                    llm=llm)
# prompt = hub.pull("hwchase17/openai-functions-agent")
prompt = hub.pull("hwchase17/react") # https://smith.langchain.com/hub/hwchase17/react?organizationId=8b40aa1a-1517-5571-a6d2-c117a8e9c4dd
agent = create_react_agent(llm, tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

query = """
{
	event(id: 1) {
		name
		fixtures {
			finished
			kickoffTime
			teamAName
			teamHName
			teamAScore
			teamHScore
		}
	}
}

Tell me which team scored the most goals in the last match.
"""
output_1 = agent_executor.invoke({"input": query})
print(output_1)

# output_2 = agent_executor.invoke({"input": "when you add 4  and 5 the result comes 10."})
# print(output_2)

