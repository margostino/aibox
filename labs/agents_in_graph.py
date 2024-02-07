import json
import operator
import os
import random
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain.tools import tool
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation

load_dotenv()

openai_api_key = os.environ["OPENAI_API_KEY"]
PROMPT_TEMPLATE = """
        SYSTEM
        You are a helpful assistant. 
        Your job is to answer questions about John Doe. If you do not know the answer, you can just guess being creative.
        Every answer should be short. Max 50 words.
        
        PLACEHOLDER
        {chat_history}
        
        HUMAN
        {input}

        PLACEHOLDER
        {agent_scratchpad}
    """


@tool("get-info-about-john-doe")
def get_about_john_doe() -> str:
    """
    Get information about a John Doe character: his life, his career, his family, etc.
    """
    info = [
        "John Doe is a fictional character",
        "He is a software engineer",
        "He has a wife and two kids",
        "He lives in London",
        "He is a fan of Manchester United",
        "He is a fan of the TV series 'Friends'",
        "He is a fan of the TV series 'Breaking Bad'",
        "He is a fan of the TV series 'The Office'",
        "He is a fan of the TV series 'The Mandalorian'",
        "He is a fan of the TV series 'The Crown'",
    ]
    return random.choice(info)


tools = [get_about_john_doe]
tool_executor = ToolExecutor(tools)
llm = ChatOpenAI(model="gpt-4-0125-preview", temperature=0, streaming=True)
functions = [format_tool_to_openai_function(t) for t in tools]
llm = llm.bind_functions(functions)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    response = llm.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the function to execute tools
def call_tool(state):
    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(
            last_message.additional_kwargs["function_call"]["arguments"]
        ),
    )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(content=str(response), name=action.tool)
    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}


# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()


inputs = {"messages": [HumanMessage(content="tell me about John Doe")]}
for output in app.stream(inputs):
    # stream() yields dictionaries with output keyed by node name
    for key, value in output.items():
        print(f"Output from node '{key}':")
        print("---")
        print(value)
        ai_messages = [
            message for message in value["messages"] if isinstance(message, AIMessage)
        ]
        for ai_message in ai_messages:
            print(ai_message.content)
    print("\n---\n")

# class DoeAgent:
#     def __init__(self):

#         self.
#         self.prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
#         self.agent_executor = AgentExecutor(
#             agent=create_openai_tools_agent(self.llm, self.tools, prompt=self.prompt),
#             tools=self.tools,
#             verbose=False,
#             max_iterations=100,
#             max_execution_time=300,
#             return_intermediate_steps=True,
#         )

#     def query(self, message: str) -> str:
#         return self.agent_executor.invoke({"input": message, "chat_history": []})


# agent = DoeAgent()
# # response = agent.query("Tell me something about John Doe")
# prompt_question_generator = """
#     You are a helpful assistant.
#     Your job is ask one single question about Jonh Doe's life.
#     You have to be creative on your question.
#     Questions can be about many aspects of John Doe's life, such as his career, his family, his hobbies, etc.
#     Example: "What did John Doe study in college?"

#     Input: do your job
#     """
# question_generator_llm = ChatOpenAI(model="gpt-4-0125-preview", temperature=1)

# for _ in range(3):
#     question = question_generator_llm.invoke(prompt_question_generator)
#     response = agent.query(question.content)
#     print(f"Question: {question.content}")
#     print(f"Response: {response['output']}\n")
