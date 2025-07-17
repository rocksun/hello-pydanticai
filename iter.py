from pydantic_graph import End
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


async def main():
    async with agent.iter('What is the capital of France?') as agent_run:
        node = agent_run.next_node  

        all_nodes = [node]

        # Drive the iteration manually:
        while not isinstance(node, End):  
            node = await agent_run.next(node)  
            all_nodes.append(node)  

        print(all_nodes)
        """
        [
            UserPromptNode(
                user_prompt='What is the capital of France?',
                instructions=None,
                instructions_functions=[],
                system_prompts=(),
                system_prompt_functions=[],
                system_prompt_dynamic_functions={},
            ),
            ModelRequestNode(
                request=ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='What is the capital of France?',
                            timestamp=datetime.datetime(...),
                        )
                    ]
                )
            ),
            CallToolsNode(
                model_response=ModelResponse(
                    parts=[TextPart(content='Paris')],
                    usage=Usage(
                        requests=1,
                        request_tokens=56,
                        response_tokens=1,
                        total_tokens=57,
                    ),
                    model_name='gpt-4o',
                    timestamp=datetime.datetime(...),
                )
            ),
            End(data=FinalResult(output='Paris')),
        ]
        """

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())