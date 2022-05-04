from acrobat import Acrobat
from agent import Agent
from yaml import safe_load


def main():
    with open("config.yaml") as f:
        config = safe_load(f)
    acrobat = Acrobat()
    agent = Agent(acrobat, **config)
    agent.train()


if __name__ == '__main__':
    main()
