from pydantic_ai import Agent
import os
import httpx
from pydantic_ai.models.gemini import GeminiModel  # Update import to use GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from dataclasses import dataclass
from pydantic_graph import BaseNode, End, Graph, GraphRunContext
from pydantic_ai.mcp import MCPServerStdio
from pydantic import BaseModel, Field
from typing import Literal

proxy = os.getenv('HTTP_PROXY')

client = httpx.AsyncClient(proxy=proxy)  # Create an HTTP client with proxy
model = GeminiModel(  # Use GeminiModel instead of OpenAIModel
    model_name='gemini-2.0-flash',
    provider=GoogleGLAProvider(
        http_client=client  # Pass the HTTP client with proxy to the provider
    )
)

@dataclass
class SimpleChat(BaseNode):
    text: str

    async def run(
        self,
        ctx: GraphRunContext,
    ) -> End[str]:
        agent = Agent(
            model=model,
            system_prompt='Be concise, reply with one sentence.' # Read system prompt from .env or use default
        )
        
        response = await agent.run(self.text)
        return End(response.output)

@dataclass
class DBQuery(BaseNode):
    text: str

    async def run(
        self,
        ctx: GraphRunContext,
    ) -> End[str]:
        
        server = MCPServerStdio(  
            'toolbox',
            args=[
                "--tools-file",
                "D:\\learns\\ai\\toolkit-mcp\\mysql.yaml",
                "--stdio"
            ]
        )
        agent = Agent(
            model=model,
            system_prompt='Please answer the question based on the provided database information. If you need to use a tool, do so.', 
            mcp_servers=[server]
        )

        async with agent.run_mcp_servers():        
            response = await agent.run(self.text)
            return End(response.output)


@dataclass
class IntentChoice(BaseModel):
    intent: Literal["SimpleChat", "DBQuery"] = Field(
        description="""Determine the user's intent.
- Use 'DBQuery' for questions about data, people, databases, or specific entities that might be in a database.
- Use 'SimpleChat' for general conversation, greetings, or questions that do not require looking up information."""
    )

@dataclass
class Intent(BaseNode):
    text: str

    async def run(
        self,
        ctx: GraphRunContext,
    ) -> SimpleChat | DBQuery:
        intent_agent = Agent(
            model=model,
            output_type=IntentChoice,
            system_prompt="You are an intent classifier. Analyze the user's text and decide whether it's a simple chat or a database query. The database contains MySQL Database Related Data."
        )
        
        print(f"Classifying intent for: '{self.text}'")
        response = await intent_agent.run(self.text)

        if response.output.intent == "DBQuery":
            print("Intent classified as: DBQuery")
            return DBQuery(text=self.text)
        else:
            print("Intent classified as: SimpleChat")
            return SimpleChat(text=self.text)

ops_helper_graph = Graph(nodes=[Intent, SimpleChat, DBQuery])

async def main(): 
    print(f"--- Running Test 1: Simple Chat ---")
    chat_question = "What is the capital of France?"
    result_chat = await ops_helper_graph.run(Intent(text=chat_question))
    print(f"Final Answer: {result_chat.output}")
    print("-" * 30)

    print(f"--- Running Test 2: DB Chat ---")
    chat_question = "列出所有的数据库"
    result_chat = await ops_helper_graph.run(Intent(text=chat_question))
    print(f"Final Answer: {result_chat.output}")
    print("-" * 30)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())