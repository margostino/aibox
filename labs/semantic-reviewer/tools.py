import json
import os

import stashy
from config import Config
from git import Repo
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from stashy.repos import Repository
from utils import normalize_review, sanitize_show_by_files

config = Config()
model, temperature = config.get_openai()
repositories = config.get_all_repository_names()
contexts = config.get_contexts()
schemas = config.get_schemas()
objetive = config.get_objective()
rules = config.get_rules()
llm = ChatOpenAI(model=model, temperature=temperature)


class PullRequest(BaseModel):
    repository: str = Field(
        description=f"should be a repository name in {repositories}"
    )
    branch: str = Field(description="should be a branch name")
    changes: list[dict] = Field(
        description="should be a list of changes (git show output) in the pull request"
    )


class Branch(BaseModel):
    repository_name: str = Field(
        description=f"should be a repository name in {repositories}"
    )
    branch: str = Field(description="should be a branch name")


reviewer_context_prompt_template = f"""
You are the owner of {len(repositories)} repositories: {repositories}.
Context about both repositories:
{contexts}
 
"""

reviewer_schemas_prompt_template = schemas
reviewer_rules_prompt_template = f"""
RULES:
- Ignore identical content. In this case you must return only an empty JSON object and nothing outside of JSON curly braces 
- Ignore formatting changing content. In this case you must return an empty JSON object and nothing outside of JSON curly braces
{rules}
- The suggestion must be short, concise and summarize if needed. Max 200 words.
- If there is same or similar change in different attributes, you must return only one suggestion for all of them.
- DO NOT return anything in your response outside of JSON curly braces
"""
reviewer_objetive_prompt_template = f"""
OBJETIVE: You have to compare and suggest a review for the changes in each repository based on the GIT SHOW. 
{objetive}
"""

reviewer_io_prompt_templates = """
REPOSITORY: {repository}
GIT SHOW:
```
{git_show}
```

Provide your response as a JSON object with the following schema:
```
{{
    "file_name": {{
        "path_to_attribute.attribute_name": {{
            "review_suggestion": "string"
        }}
    }}
}}
```

Where:
1. "review_suggestion" is a text of max 200 words with change review suggested
"""

reviewer_prompt_template = f"""
{reviewer_context_prompt_template}
{reviewer_schemas_prompt_template}
{reviewer_objetive_prompt_template}
{reviewer_io_prompt_templates}
{reviewer_rules_prompt_template}
"""

reviewer_chain = LLMChain(
    llm=llm, prompt=PromptTemplate.from_template(reviewer_prompt_template)
)


def fetch_branches_for_open_pull_requests(repository_name: str) -> list:
    repository = config.get_repository(repository_name)

    if not repository["enabled"]:
        return []

    username = os.getenv("REPO_USERNAME")
    password = os.getenv("REPO_PASSWORD")
    local_repo_cache_path = f"{config.get_cache_path()}/{repository_name}"

    Repo.clone_from(repository["ssh_endpoint"], to_path=local_repo_cache_path)

    if len(repository["branches"]) > 0:
        return repository["branches"]

    stash = stashy.connect(repository["http_endpoint"], username, password)
    repo: Repository = stash.projects[repository["project"]].repos[repository_name]

    branches = []
    for pr in repo.pull_requests.all():
        state = pr["state"]
        from_branch = pr["fromRef"]["displayId"]
        if state == "OPEN":
            branches.append(from_branch)

    return branches


def review_semantics(repository_name: str, branch: str) -> list[dict]:
    repository = config.get_repository(repository_name)
    prefix_file = repository["prefix_file"]
    local_repo_cache_path = f"{config.get_cache_path()}/{repository_name}"
    local_repo = Repo(local_repo_cache_path)

    branch_commits = [
        commit
        for commit in local_repo.iter_commits(
            rev=f"master..origin/{branch}", max_count=10
        )
    ]

    changes = [
        sanitize_show_by_files(local_repo.git.show(commit.hexsha), prefix_file)
        for commit in branch_commits
    ]
    changes = [change for change in changes if change]

    reviews = []
    for change in changes:
        try:
            review = reviewer_chain.run(repository=repository, git_show=change)
            parsed_review = normalize_review(review)
        except Exception as e:
            print(
                f"Unable to get show for branch {branch}, repository {repository}: {e}"
            )
            parsed_review = {"error": str(e)}

        reviews.append(parsed_review)

    if len(reviews) == 0:
        reviews.append(
            {
                "skipped": f"pull request in repository {repository} with branch {branch} has no configuration changes (YAML files)"
            }
        )

    for review in reviews:
        print(json.dumps(review, indent=4, sort_keys=True))

    return reviews


pull_requests_fetcher_tool = StructuredTool.from_function(
    func=fetch_branches_for_open_pull_requests,
    name="BranchesFetcher",
    description=f"Tool that fetch branches of the open pull requests for a given repository in {repositories} and return the list of branches",
)

semantic_reviewer_tool = StructuredTool.from_function(
    func=review_semantics,
    name="SemanticReviewer",
    args_schema=Branch,
    description="Tool that take a branch for a given repository, get the GIT SHOW between every branch commit and master and make reviews and suggestions",
)
