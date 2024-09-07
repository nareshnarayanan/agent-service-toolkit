from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
import httpx, os

class AgentRegistry:
    def __init__(self):
        self._agents = {}

    def register(self, name, agent):
        self._agents[name] = agent

    def get(self, name):
        return self._agents.get(name)

    def get_all(self):
        return self._agents.values()

    def get_all_names(self):
        return self._agents.keys()

    def __iter__(self):
        return iter(self._agents.values())

    def __len__(self):
        return len(self._agents)

    def __contains__(self, name):
        return name in self._agents

    def __getitem__(self, name):
        return self._agents[name]

    def __setitem__(self, name, agent):
        self._agents[name] = agent

    def __delitem__(self, name):
        del self._agents[name]

def get_llm() -> BaseChatModel:
    return models["gpt-4o-mini"]


models = {
    "llama3.1-70b-cerebras": ChatOpenAI(
        model="llama3.1-70b",
        api_key=os.getenv("CEREBRAS_API_KEY"),
        base_url="https://api.cerebras.ai/v1",
        # stream=True
        ),
    "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini", streaming=True,
                              http_client=httpx.Client(http2=True)),
    "llama-3.1-70b": ChatGroq(model="llama-3.1-70b-versatile", temperature=0.5),
}
