import os

from agent import Agent
from config import Config


def main():
    config = Config()
    cache_path = config.get_cache_path()
    init(cache_path)

    prompt = config.get_prompt()
    agent = Agent(config)
    print(agent.run(prompt))

    cleanup(cache_path)


def init(cache_path: str):
    os.system(f"rm -rf {cache_path}")
    os.system(f"mkdir {cache_path}")


def cleanup(cache_path: str):
    os.system(f"rm -rf {cache_path}")


if __name__ == '__main__':
    main()
