from pydantic_ai import Agent
import os
import httpx
from pydantic_ai.models.gemini import GeminiModel  # Update import to use GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider

http_proxy = os.getenv('HTTP_PROXY')
# https_proxy = os.getenv('HTTPS_PROXY')
client = httpx.AsyncClient(proxy=http_proxy, timeout=120)

model = GeminiModel( 
    model_name='gemini-2.0-flash',
    provider=GoogleGLAProvider(
        http_client=client
    )
)

agent = Agent(model=model)

# result_sync = agent.run_sync('What is the capital of Italy?')
# print(result_sync.output)
# > Rome


async def main():
    nodes = []
    # Begin an AgentRun, which is an async-iterable over the nodes of the agent's graph
    async with agent.iter('What is the capital of France?') as agent_run:
        async for node in agent_run:
            # Each node represents a step in the agent's execution
            print(node)
            nodes.append(node)
    print(nodes)

    print(agent_run.result.output)
    # result = await agent.run('What is the capital of France?')
    # print(result.output)
    # #> Paris

    # async with agent.run_stream('What is the capital of the UK?') as response:
    #     print(await response.get_output())
    #     #> London

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())