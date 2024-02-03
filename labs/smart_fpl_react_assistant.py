import os
from typing import Any

import requests
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain.prompts import PromptTemplate
from langchain.tools import StructuredTool, tool
from langchain_openai import OpenAI
from pydantic import BaseModel, Field

GRAPHQL_ENDPOINT = "https://anfield-api-margostino.vercel.app/api/query"
EVENTS_QUERY = """
query Event($id: Int!) {
    event(id: $id) {
        name
        fixtures {
            finished
            id
            event
            kickoffTime
            teamAName
            teamAScore
            teamHName
            teamHScore
        }
    }
}
"""

load_dotenv()

openai_api_key = os.environ["OPENAI_API_KEY"]


class GameweekId(BaseModel):
    gameweek_id: int = Field(description="should be and ID between 1 and 38")


@tool("gameweek-tool", args_schema=GameweekId, return_direct=True)
def get_gameweek_by_id(gameweek_id: int) -> str:
    """
    Get information about a specific Premier League Gameweek by ID.
    This returns all results for a given Premier League Gameweek.
    """
    variables = {"id": gameweek_id}
    res = requests.post(
        GRAPHQL_ENDPOINT,
        json={"query": EVENTS_QUERY, "variables": variables},
        timeout=10,
    )
    if res.status_code != 200:
        print(f"Failure querying events. Error: {res.status_code} - {res.content}")
    # return json.dumps(response.json(), indent=2)
    # fixtures = res.json()["data"]["event"]["fixtures"]
    output = """
        - Manchester United vs Liverpool: 3-10
        - Arsenal vs Chelsea: 1-2
    """
    return output
    # return fixtures


# llm = OpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-instruct")
llm = OpenAI(temperature=0)
# get_gameweek_by_id_tool = StructuredTool.from_function(
#     name="GetGameweekById",
#     func=get_gameweek_by_id,
#     args_schema=GameweekId,
#     description="useful for searching information about a specific Gameweek. It returns a dictionary with all fixtures with results for a given Gameweek",
# )
from langchain.chains import LLMChain

prompt_template_for_aggs_calculator = PromptTemplate.from_template(
    """
    You are a helpful assistant.
    Question: {input}
    """
)
llm2 = OpenAI(model="gpt-4-0125-preview")
llm_chain = LLMChain(llm=llm2, prompt=prompt_template_for_aggs_calculator)
aggs_calculator = StructuredTool.from_function(
    func=llm_chain.run,
    name="LLMChain",
    description="helpful assistant",
)

prefined_tools = load_tools(
    ["graphql"],
    graphql_endpoint="https://anfield-api-margostino.vercel.app/api/query",
    llm=llm2,
)
prefined_tools = load_tools(
    ["llm-math"],
    llm=llm2,
)
custom_tools = [get_gameweek_by_id, aggs_calculator]
# tools = prefined_tools + custom_tools
tools = custom_tools
prompt = hub.pull("hwchase17/react")
prompt_template = PromptTemplate.from_template(
    """
        You are a helpful assistant. Your job is to answer questions as the best you can about Premier League Gameweeks (season 2023/24). 

        You should stop when you know the answer to the question.

        You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought:{agent_scratchpad}
    """
)

agent = create_react_agent(llm, tools, prompt=prompt_template)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)
# response = agent_executor.invoke({"input": "Which team was the best in Gameweek ID 1?"})
# response = agent_executor.invoke({"input": "How many fixtures are in Gameweek 1?"})
# "Which team scored most goals in Gameweek 1 ?"
# "input": "How many fixtures are in Gameweek 1 ?",
# from langchain.agents.format_scratchpad.openai_tools import (
#     format_to_openai_tool_messages,
# )

# response = agent_executor.invoke(
#     {
#         "input": "In Gameweek 1, which team scored goals the most?",
#         "agent_scratchpad": lambda x: format_to_openai_tool_messages(
#             x["intermediate_steps"]
#         ),
#     }
# )
response = agent_executor.invoke(
    {
        "input": "In Gameweek 1, which team scored goals the most?",
        "agent_scratchpad": "",
    }
)
print(response)

# def generate_response(input_query):
#     tools = [get_gameweek_by_id]
#     llm = OpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-instruct")
#     prompt = hub.pull("hwchase17/react")
#     st_callback = StreamlitCallbackHandler(st.container())
#     agent = create_react_agent(llm, tools, prompt=prompt)
#     agent_executor = AgentExecutor(
#         agent=agent, tools=tools, verbose=True, callbacks=[st_callback]
#     )
#     response = agent_executor.invoke({"input": input_query})
#     return st.success(response)

# st.set_page_config(page_title="🦜🔗 Ask about Fantasy PL")
# st.title("🦜🔗 Ask about Fantasy PL")

# # Input widgets
# question_list = [
#     "Which team was the best in Gameweek 1?",
#     "How many goals have been scored in Gameweek 2?",
#     "Other",
# ]
# query_text = st.selectbox("Select an example query:", question_list)
# openai_api_key = st.text_input(
#     "OpenAI API Key", type="password", disabled=not (query_text)
# )

# # App logic
# if query_text == "Other":
#     query_text = st.text_input(
#         "Enter your query:",
#         placeholder="Enter query here ...",
#     )
# if not openai_api_key.startswith("sk-"):
#     st.warning("Please enter your OpenAI API key!", icon="⚠")
# if openai_api_key.startswith("sk-"):
#     st.header("Output")
#     generate_response(query_text)
