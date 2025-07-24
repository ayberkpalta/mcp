"""
Advanced MCP Agent Usage Example
This file demonstrates how to build your own agents.
"""

import asyncio
import time
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

# Import custom functions
from custom_functions import (
    divide_numbers, power_numbers, factorial, 
    calculate_area_circle, fibonacci,
    reverse_string, count_words, to_uppercase
)

# Built-in functions
def add_numbers(a: int, b: int) -> int:
    """Adds two numbers."""
    print(f"Math expert is adding {a} and {b}")
    return a + b

def multiply_numbers(a: int, b: int) -> int:
    """Multiplies two numbers."""
    print(f"Math expert is multiplying {a} and {b}")
    return a * b

# Initialize MCP App
app = MCPApp(
    name="advanced_mcp_agent",
    settings=".mcp_agent.config.yaml"
)

async def advanced_example():
    """Advanced usage example of MCP Agent"""
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context
        
        # 1. Math Agent - Handles all math functions
        math_agent = Agent(
            name="advanced_math_agent",
            instruction="""You are an advanced math expert.
            You have access to addition, multiplication, division, exponentiation,
            factorial, fibonacci, and circle area calculation functions.
            Analyze the user's request and use the appropriate function.""",
            functions=[
                add_numbers, multiply_numbers, divide_numbers, 
                power_numbers, factorial, fibonacci, calculate_area_circle
            ],
        )
        
        # 2. String Agent - Handles text operations
        string_agent = Agent(
            name="string_agent", 
            instruction="""You are a text processing expert.
            You have access to reverse string, count words,
            and convert to uppercase functions.""",
            functions=[reverse_string, count_words, to_uppercase],
        )
        
        # Test the Math Agent
        async with math_agent:
            logger.info("=== MATH AGENT TEST ===")
            
            llm = await math_agent.attach_llm(OpenAIAugmentedLLM)
            
            test_queries = [
                "Add 5 and 3, then multiply the result by 2",
                "Divide 10 by 3",
                "Calculate 2 raised to the power of 8", 
                "Find the factorial of 5",
                "Find the 10th number in the Fibonacci sequence",
                "Calculate the area of a circle with radius 5"
            ]
            
            for query in test_queries:
                try:
                    logger.info(f"Query: {query}")
                    result = await llm.generate_str(message=query)
                    logger.info(f"Answer: {result}")
                    print("-" * 50)
                except Exception as e:
                    logger.error(f"Error: {e}")
        
        # Test the String Agent  
        async with string_agent:
            logger.info("=== STRING AGENT TEST ===")
            
            llm = await string_agent.attach_llm(OpenAIAugmentedLLM)
            
            string_queries = [
                "Reverse this text: 'Hello World'",
                "Count the number of words in this sentence: 'I am developing an MCP agent with Python'",
                "Convert this text to uppercase: 'mcp agent is very powerful'"
            ]
            
            for query in string_queries:
                try:
                    logger.info(f"Query: {query}")
                    result = await llm.generate_str(message=query)
                    logger.info(f"Answer: {result}")
                    print("-" * 50)
                except Exception as e:
                    logger.error(f"Error: {e}")

if __name__ == "__main__":
    print("🚀 Launching Advanced MCP Agent Example...")
    start = time.time()
    asyncio.run(advanced_example())
    end = time.time()
    print(f"✅ Total runtime: {end - start:.2f} seconds")
