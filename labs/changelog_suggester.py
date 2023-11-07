from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool

load_dotenv()

tool_description = """
    Tool that read logs of the GIT commit ID from the repository and return the text of it.
    The format of this file is the following:
    ```AuthorName CommitId CommitComment
    Number Number FileName```

    For example:
    ```
    Martin 1234567890 Added new feature
    1 1 src/main.py
    2 2 src/utils.py
    Martin 1234567890 Update documentation
    1 1 README.md
    ```
    """


def get_logs_from_git_commit(commit_id: str) -> int:
    """
    Tool that read logs of the GIT commit ID from the repository and return the text of it.
    The format of this file is the following:
    ```AuthorName CommitId CommitComment
    Number Number FileName```

    For example:
    ```
    Martin 1234567890 Added new feature
    1 1 src/main.py
    2 2 src/utils.py
    Martin 1234567890 Update documentation
    1 1 README.md
    ```
    """
    with open("git_log.txt", "r") as f:
        data = f.read()
        return data


# llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)

prompt_template = "Create a CHANGELOG.md from the following Git LOGS content: {content}."
sw_engineer_llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)

# TOOLS
changlog_tool_description = """
                            The format of the CHANGELOG must be like the following: \n
                            FORMAT:
                                ```
                                ## [$SEMANTIC_VERSION] - $DATE
                                
                                ### Added | Changed | Deprecated | Removed | Fixed | Security
                                - $DESCRIPTION_1
                                - $DESCRIPTION_2
                                
                                ### Added | Changed | Deprecated | Removed | Fixed | Security
                                - $DESCRIPTION_3
                                
                                ### Added | Changed | Deprecated | Removed | Fixed | Security
                                - $DESCRIPTION_4
                                ```
                            
                            EXAMPLE:
                                ```
                                ## [0.0.1] - 2023-09-20
                                
                                ### Added
                                - Feature 1 to do something
                                - Feature 2 to do something
                                
                                ### Changed
                                - Something changed in the code
                                
                                ### Fixed
                                - Bug fix in the code
                                ```
                        """
changelog_creator_tool = Tool.from_function(
    func=sw_engineer_llm_chain.run,
    name="ChangelogCreator",
    description=changlog_tool_description
)

commit_logs_fetcher_tool = Tool.from_function(
    func=get_logs_from_git_commit,
    name="LogsGitCommit",
    description=tool_description
)

tools = [commit_logs_fetcher_tool, changelog_creator_tool]

agent = initialize_agent(
    tools=tools,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=True,
    max_execution_time=60,
    early_stopping_method="generate"
)

prompt = f"Use your tools to get the LOGS from the GIT commit ID a323434343 and create a CHANGELOG.md. " \
         f"Use file names as inspiration to understand semantically the logs BUT do not include the filename."

print(agent.run(prompt))
