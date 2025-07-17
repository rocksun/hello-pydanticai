from pydantic_ai import Agent, RunContext
import os
import httpx
from pydantic_ai.models.gemini import GeminiModel  # Update import to use GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic import BaseModel
from dataclasses import dataclass
import readability
from markdownify import markdownify as md

proxy = os.getenv('HTTP_PROXY')
client = httpx.AsyncClient(proxy=proxy, timeout=120)

model = GeminiModel( 
    model_name='gemini-2.0-flash',
    provider=GoogleGLAProvider(
        http_client=client
    )
)

@dataclass
class MyDeps:  
    url: str


class ArticleMeta(BaseModel):
    url: str
    summary: str

meta_agent = Agent(
    model=model,
    retries=3,
    deps_type=MyDeps,
    output_type=ArticleMeta,
    instructions="""Analyze the content and provide structured response with Chinese."""
)

translator_agent = Agent(
    model=model,
    retries=3,
    instructions="""As a cloud native expert and translator.
Fetch and Translate this article from English to Chinese. 
Keep the markdown format intact.

TRANSLATION REQUIREMENTS:

- Do not translate names of non-famous people.
- Do not translate the text in the code block.
- Do not print explanation, just print the translation.
- Ensure the text of link will be translated.
- Translate 'obserablity' into '可观测性'.
- Make sure translate the text into Simplified Chinese."""
)


@translator_agent.tool_plain
async def fetch_and_convert_to_markdown(url: str) -> str:
    """
    Fetches and converts the content of the given URL to markdown format.

    Args:
        url (str): The URL to fetch content from.

    Returns:
        str: The content converted to markdown format.
    """
    response = await client.get(url)  # Fetch the URL content
    response.raise_for_status()
    original_html = response.text
    doc = readability.Document(original_html)
    main_html = doc.summary()

    markdown = md(main_html, strip=['img', 'script', 'style'])  # Convert HTML to markdown
    return markdown

async def main():
    url = 'https://thenewstack.io/boost-performance-with-react-server-components-and-next-js/'
    # print(content)
    result = await translator_agent.run(url)
    print(result.output)
    meta = await meta_agent.run(result.output, deps=MyDeps(url=url))
    print(meta)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
