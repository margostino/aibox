import operator
import os
import random
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

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


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


class DoeAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-0125-preview", temperature=0)
        self.tools = [self.about_john_doe]
        self.prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        self.agent_executor = AgentExecutor(
            agent=create_openai_tools_agent(self.llm, self.tools, prompt=self.prompt),
            tools=self.tools,
            verbose=False,
            max_iterations=100,
            max_execution_time=300,
            return_intermediate_steps=True,
        )

    def query(self, message: str) -> str:
        return self.agent_executor.invoke({"input": message, "chat_history": []})

    @staticmethod
    @tool("get-info-about-john-doe")
    def about_john_doe() -> str:
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


agent = DoeAgent()
# response = agent.query("Tell me something about John Doe")
prompt_question_generator = """
    You are a helpful assistant.
    Your job is ask one single question about Jonh Doe's life.
    You have to be creative on your question.
    Questions can be about many aspects of John Doe's life, such as his career, his family, his hobbies, etc.
    Example: "What did John Doe study in college?"

    Input: do your job
    """
question_generator_llm = ChatOpenAI(model="gpt-4-0125-preview", temperature=1)

for _ in range(3):
    question = question_generator_llm.invoke(prompt_question_generator)
    response = agent.query(question.content)
    print(f"Question: {question.content}")
    print(f"Response: {response['output']}\n")
