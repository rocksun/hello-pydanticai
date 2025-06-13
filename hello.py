from pydantic_ai import Agent
import os
import httpx
from pydantic_ai.models.gemini import GeminiModel  # Update import to use GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider

proxy = os.getenv('HTTP_PROXY')

client = httpx.AsyncClient(proxy=proxy)  # Create an HTTP client with proxy
model = GeminiModel(  # Use GeminiModel instead of OpenAIModel
    model_name='gemini-2.0-flash',
    provider=GoogleGLAProvider(
        http_client=client  # Pass the HTTP client with proxy to the provider
    )
)

agent = Agent(
    model=model,
    system_prompt='Be concise, reply with one sentence.' # Read system prompt from .env or use default
)

# Run the agent synchronously, conducting a conversation with the LLM.
# Here the exchange should be very short: PydanticAI will send the system prompt and the user query to the LLM,
# the model will return a text response. See below for a more complex run.
result = agent.run_sync('Where does "hello world" come from?')
print(result.output)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""