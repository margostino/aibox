from typing import Any

import toml
from dotenv import load_dotenv


class Config:
    def __init__(self):
        load_dotenv()
        self.config = toml.load('config.toml')

    def get_openai(self) -> (str, float):
        model = self._get_config_value('openai', 'model')
        temperature = self._get_config_value('openai', 'temperature')
        return model, temperature

    def get_agent(self) -> dict:
        return self.config['agent']

    def get_prompt(self) -> str:
        return self.config['llm']['prompt']

    def get_repository(self, repository_name: str) -> dict:
        filtered_repositories = [repository for repository in self.config['repositories'] if repository['name'] == repository_name]

        if len(filtered_repositories) > 0:
            return filtered_repositories[0]
        else:
            raise Exception(f"Repository {repository_name} not found")

    def get_cache_path(self):
        return self._get_config_value("cache", "path")

    def get_cache_paths(self):
        return [repository['local_cache_path'] for repository in self.config['repositories']]

    def get_all_repository_names(self):
        return [repository['name'] for repository in self.config['repositories']]

    def get_contexts(self):
        return "\n".join([repository['context'] for repository in self.config['repositories']])

    def get_schemas(self):
        return "\n".join([repository['schema'] for repository in self.config['repositories']])

    def get_objective(self):
        return "\n".join([repository['objective'] for repository in self.config['repositories']])

    def get_rules(self):
        return "\n".join([repository['rules'] for repository in self.config['repositories']])

    def _get_config_value(self, *keys: str) -> Any:
        value = self.config
        for key in keys:
            value = value.get(key)
        return value
