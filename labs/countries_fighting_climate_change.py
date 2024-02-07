import functools
import json
import operator
from email import message
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import StructuredTool
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

MODEL = "gpt-4-0125-preview"


def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    functions = [format_tool_to_openai_function(t) for t in tools]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_functions(functions)


def get_country_prompt_for(country_name: str) -> PromptTemplate:
    return PromptTemplate.from_template(
        f"""
        You are representative of {country_name}. 
        Your job is to establish communication with other countries and agree on an action plan to fight Climate Change.    
        Keep your answers short and to the point: Max 50 words.
        
        You finish when you and other countries have an bilateral agreement to fight Climate Change. The agreement should include:
            - Objetives: what you want to achieve
            - Action plan: how you will achieve your objetives
            - Resources: what you need to achieve your objetives
            - Timeline: when you will achieve your objetives
            
        In order to achieve your goals:
         - you have to be collaborative with other countries and help them out with resources and services.        
         - you have to ask other countries for help.
        
        Input: {{input}}
        """
    )


argentina_prompt = get_country_prompt_for("Argentina")
sweden_prompt = get_country_prompt_for("Sweden")

argentina_llm_chain = LLMChain(llm=ChatOpenAI(model=MODEL), prompt=argentina_prompt)
sweden_llm_chain = LLMChain(llm=ChatOpenAI(model=MODEL), prompt=sweden_prompt)


class CountryInput(BaseModel):
    input: str = Field(
        description="should be the message from a country representative"
    )


argentina_tool = StructuredTool.from_function(
    func=argentina_llm_chain.run,
    name="ArgentinaRepresentative",
    args_schema=CountryInput,
    description="useful tool to interact with a representative of Argentina",
)

sweden_tool = StructuredTool.from_function(
    func=sweden_llm_chain.run,
    name="SwedenRepresentative",
    args_schema=CountryInput,
    description="useful tool to interact with a representative of Sweden",
)


# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


# Helper function to create a node for a given agent
def agent_node(state, agent, name):
    result = agent.invoke(state)
    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, FunctionMessage):
        pass
    else:
        result = HumanMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name,
    }


llm = ChatOpenAI(model=MODEL)

# Argentina
argentina_agent = create_agent(
    llm,
    [argentina_tool],
    system_message="You should reach an agreement with Sweden in order to establish a action plan to fight Climate Change.",
)
argentina_node = functools.partial(agent_node, agent=argentina_agent, name="Argentina")

# Sweden
sweden_agent = create_agent(
    llm,
    [sweden_tool],
    system_message="You should reach an agreement with Argentina in order to establish a action plan to fight Climate Change.",
)
sweden_node = functools.partial(agent_node, agent=sweden_agent, name="Sweden")

tools = [argentina_tool, sweden_tool]
tool_executor = ToolExecutor(tools)


def tool_node(state):
    """This runs tools in the graph

    It takes in an agent action and calls that tool and returns the result."""
    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    tool_input = json.loads(
        last_message.additional_kwargs["function_call"]["arguments"]
    )
    # We can pass single-arg inputs by value
    if len(tool_input) == 1 and "__arg1" in tool_input:
        tool_input = next(iter(tool_input.values()))
    tool_name = last_message.additional_kwargs["function_call"]["name"]
    action = ToolInvocation(
        tool=tool_name,
        tool_input=tool_input,
    )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(
        content=f"{tool_name} response: {str(response)}", name=action.tool
    )
    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}


# Either agent can decide to end
def router(state):
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if "function_call" in last_message.additional_kwargs:
        # The previus agent is invoking a tool
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return "end"
    return "continue"


workflow = StateGraph(AgentState)

workflow.add_node("Argentina", argentina_node)
workflow.add_node("Sweden", sweden_node)
workflow.add_node("call_tool", tool_node)

workflow.add_conditional_edges(
    "Argentina",
    router,
    {"continue": "Sweden", "call_tool": "call_tool", "end": END},
)
workflow.add_conditional_edges(
    "Sweden",
    router,
    {"continue": "Argentina", "call_tool": "call_tool", "end": END},
)

workflow.add_conditional_edges(
    "call_tool",
    # Each agent node updates the 'sender' field
    # the tool calling node does not, meaning
    # this edge will route back to the original agent
    # who invoked the tool
    lambda x: x["sender"],
    {
        "Argentina": "Argentina",
        "Sweden": "Sweden",
    },
)
workflow.set_entry_point("Argentina")
graph = workflow.compile()

for s in graph.stream(
    {
        "messages": [
            HumanMessage(
                content="Come up with an action plan for renewable energy to fight Climate Change?"
            )
        ],
    },
    # Maximum number of steps to take in the graph
    {"recursion_limit": 150},
):
    for agent_key, agent_state in s.items():
        for message in agent_state["messages"]:
            # check if 'name' is present in the message
            if hasattr(message, "name"):
                print(f"{message.name} ({message.type}): {message.content}")
            else:
                print(f"{message.type}: {message.content}")
    print("----")
