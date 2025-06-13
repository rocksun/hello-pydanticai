from pydantic_ai import Agent
import os
import httpx
from pydantic_ai.models.gemini import GeminiModel  # Update import to use GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from markitdown import MarkItDown

proxy = os.getenv('HTTP_PROXY')
client = httpx.AsyncClient(proxy=proxy, timeout=120)

model = GeminiModel( 
    model_name='gemini-2.0-flash',
    provider=GoogleGLAProvider(
        http_client=client
    )
)

translator_agent = Agent(
    model=model,
    retries=3,
    instructions="""As a cloud native expert and translator.
Translate this article from English to Chinese. 
Keep the markdown format intact.

TRANSLATION REQUIREMENTS:

- Do not translate names of non-famous people.
- Do not translate the text in the code block.
- Do not print explanation, just print the translation.
- Ensure the text of link will be translated.
- Translate 'obserablity' into '可观测性'.
- Make sure translate the text into Simplified Chinese."""
)

async def fetch_and_convert_to_markdown(url: str) -> str:
    """
    Fetches the content of the given URL and converts it to markdown format.
    
    Args:
        url (str): The URL to fetch content from.
    
    Returns:
        str: The content converted to markdown format.
    """
    async with httpx.AsyncClient() as client:
        md = MarkItDown()
        markdown_content = md.convert(url)
        return str(markdown_content)



async def main():
    content = await fetch_and_convert_to_markdown('https://github.com/microsoft/markitdown')
    # print(content)
    result = await translator_agent.run(content)
    print(result.output)
    pass

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
