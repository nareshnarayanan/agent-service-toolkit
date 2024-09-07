from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
import httpx, os, logging
import importlib.util

logger = logging.getLogger(__name__)

class AgentRegistry:
    def __init__(self, load_direrctory=None):
        self._agents = {}

        if load_direrctory:
            # Find all files in the directory
            logger.info(f"Loading agents from {load_direrctory}")
            # Get absolute path from relative directory
            abs_directory = os.path.abspath(load_direrctory)

            # Iterate through all files in the directory
            for filename in os.listdir(abs_directory):
                # Filter only Python files (ending with .py)
                if filename.endswith('.py') and filename != '__init__.py':
                    # Create module name by removing the .py extension
                    module_name = filename[:-3]
                    logger.info(f"Loading agent: {module_name}")

                    # Full path of the Python file
                    file_path = os.path.join(abs_directory, filename)

                    # Create a module spec from the file path
                    spec = importlib.util.spec_from_file_location(module_name, file_path)

                    # If the spec is found, load the module
                    if spec:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        # Make the module accessible by its name
                        globals()[module_name] = module
                        logger.info(f"Loaded module: {module_name}")
                        self.register(module_name, module.__getattribute__(module_name))
                        logger.info(f"Finished loading agent: {module_name}")
                    else:
                        logger.error(f"Could not load module {module_name}")
                else:
                    logger.warn(f"Skipping non-Python file: {filename}")

    def register(self, name, agent):
        self._agents[name] = agent

    def get(self, name):
        return self._agents.get(name)
    
    def get_graph(self, name):
        return self._agents.get(name).get_graph(xray=True).draw_mermaid()

    def get_all(self):
        return self._agents.values()

    def get_all_names(self):
        return self._agents.keys()

def get_llm() -> BaseChatModel:
    return models["llama-3.1-70b"]

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
