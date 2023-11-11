from typing import Any

from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.tools import StructuredTool
from pydantic import BaseModel

from config import Config
from tools import semantic_reviewer_tool, pull_requests_fetcher_tool
from utils import _handle_error


class Agent(BaseModel):
    tools: list[StructuredTool] = []
    llm: ChatOpenAI = None
    agent: AgentExecutor = None

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model, temperature = config.get_openai()
        self.tools = [pull_requests_fetcher_tool, semantic_reviewer_tool]
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self._initialize_agent()

    def _initialize_agent(self):
        self.agent = initialize_agent(
            tools=self.tools,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            llm=self.llm,
            verbose=True,
            max_execution_time=1000,
            # early_stopping_method="generate",
            handle_parsing_errors=_handle_error,
        )

    def run(self, prompt: str) -> Any:
        return self.agent.run(prompt)
