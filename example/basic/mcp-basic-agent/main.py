import asyncio
import os
import time

from mcp_agent.app import MCPApp
from mcp_agent.config import (
    Settings,
    LoggerSettings,
    MCPSettings,
    MCPServerSettings,
    OpenAISettings,
    AnthropicSettings,
)
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.llm_selector import ModelPreferences
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM


# Settings'i programatik olarak tanımla
settings = Settings(
    execution_engine="asyncio",
    logger=LoggerSettings(type="console", level="info"),
    mcp=MCPSettings(
        servers={
            "fetch": MCPServerSettings(command="uvx", args=["mcp-server-fetch"]),
            "filesystem": MCPServerSettings(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem"]),
        }
    ),
    openai=OpenAISettings(
        api_key=".....",
        base_url="https://openrouter.ai/api/v1",
        default_model="anthropic/claude-3.5-sonnet",
    ),
    anthropic=None,  # kesinlikle None yap!
)

# MCP App'i programatik settings ile başlat
app = MCPApp(
    name="basic_mcp_agent",
    settings=settings  # Config dosyası yerine programatik settings kullan
)

async def example_usage():
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context

        logger.info("Current config:", data=context.config.model_dump())

        # Mevcut dizini filesystem server'ın args'ına ekle
        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

        finder_agent = Agent(
            name="finder",
            instruction="""You are an agent with access to the filesystem, 
            as well as the ability to fetch URLs. Your job is to identify 
            the closest match to a user's request, make the appropriate tool calls, 
            and return the URI and CONTENTS of the closest match.""",
            server_names=["fetch", "filesystem"],
        )

        async with finder_agent:
            logger.info("finder: Connected to server, calling list_tools...")
            result = await finder_agent.list_tools()
            logger.info("Tools available:", data=result.model_dump())

            # İlk olarak OpenAI LLM kullan (config dosyası okuma için)
            llm = await finder_agent.attach_llm(OpenAIAugmentedLLM)
            result = await llm.generate_str(
                message="Print the contents of .mcp_agent.config.yaml verbatim",
            )
            logger.info(f".mcp_agent.config.yaml contents: {result}")

            # Web fetch için de OpenAI LLM kullan
            result = await llm.generate_str(
                message="Print the first 2 paragraphs of https://modelcontextprotocol.io/introduction",
            )
            logger.info(f"First 2 paragraphs of Model Context Protocol docs: {result}")

            # Multi-turn conversation - aynı LLM'i kullanarak devam et
            result = await llm.generate_str(
                message="Summarize those paragraphs in a 128 character tweet",
                request_params=RequestParams(
                    modelPreferences=ModelPreferences(
                        costPriority=0.1, 
                        speedPriority=0.2, 
                        intelligencePriority=0.7
                    ),
                ),
            )
            logger.info(f"Paragraph as a tweet: {result}")


if __name__ == "__main__":
    start = time.time()
    asyncio.run(example_usage())
    end = time.time()
    t = end - start

    print(f"Total run time: {t:.2f}s")
