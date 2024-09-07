
from client import AgentClient
from schema import ChatMessage

model = "llama-3.1-70b"

#### ASYNC ####
import asyncio
async def amain():
    client = AgentClient()

    print("Chat example:")
    response = await client.ainvoke("Tell me a brief joke?", model=model)
    response.pretty_print()

    print("\nStream example:")
    async for message in client.astream("Share a quick fun fact?", model=model):
        if isinstance(message, str):
            print(message, flush=True, end="|")
        elif isinstance(message, ChatMessage):
            message.pretty_print()
        else:
            print(f"ERROR: Unknown type - {type(message)}")

asyncio.run(amain())

#### SYNC ####
client = AgentClient()

print("Chat example:")
response = client.invoke("Tell me a brief joke?", model=model)
response.pretty_print()

print("\nStream example:")
for message in client.stream("Share a quick fun fact?", model=model):
    if isinstance(message, str):
        print(message, flush=True, end="|")
    elif isinstance(message, ChatMessage):
        message.pretty_print()
    else:
        print(f"ERROR: Unknown type - {type(message)}")
