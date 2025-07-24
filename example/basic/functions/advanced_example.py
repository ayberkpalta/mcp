"""
Gelişmiş MCP Agent Kullanım Örneği
Bu dosya size nasıl kendi agent'larınızı oluşturacağınızı gösterir
"""

import asyncio
import time
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

# Özel fonksiyonlarımızı import edelim
from custom_functions import (
    divide_numbers, power_numbers, factorial, 
    calculate_area_circle, fibonacci,
    reverse_string, count_words, to_uppercase
)

# Mevcut fonksiyonlar
def add_numbers(a: int, b: int) -> int:
    """Adds two numbers."""
    print(f"Math expert is adding {a} and {b}")
    return a + b

def multiply_numbers(a: int, b: int) -> int:
    """Multiplies two numbers."""
    print(f"Math expert is multiplying {a} and {b}")
    return a * b

# MCP App'i başlat
app = MCPApp(
    name="advanced_mcp_agent",
    settings=".mcp_agent.config.yaml"
)

async def advanced_example():
    """Gelişmiş MCP Agent kullanım örneği"""
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context
        
        # 1. Matematik Agent'ı - Tüm matematik fonksiyonları ile
        math_agent = Agent(
            name="advanced_math_agent",
            instruction="""Sen gelişmiş matematik uzmanısın. 
            Toplama, çarpma, bölme, üs alma, faktöriyel, 
            fibonacci ve daire alanı hesaplama fonksiyonlarına erişimin var.
            Kullanıcının isteğini analiz et ve uygun fonksiyonu kullan.""",
            functions=[
                add_numbers, multiply_numbers, divide_numbers, 
                power_numbers, factorial, fibonacci, calculate_area_circle
            ],
        )
        
        # 2. String Agent'ı - Metin işlemleri için
        string_agent = Agent(
            name="string_agent", 
            instruction="""Sen metin işleme uzmanısın.
            Metinleri tersine çevirme, kelime sayma ve büyük harfe çevirme 
            fonksiyonlarına erişimin var.""",
            functions=[reverse_string, count_words, to_uppercase],
        )
        
        # Matematik Agent'ını test et
        async with math_agent:
            logger.info("=== MATEMATİK AGENT TESTİ ===")
            
            llm = await math_agent.attach_llm(OpenAIAugmentedLLM)
            
            # Farklı matematik işlemleri test et
            test_queries = [
                "5 ile 3'ü topla, sonucu 2 ile çarp",
                "10'u 3'e böl",
                "2'nin 8'inci kuvvetini hesapla", 
                "5'in faktöriyelini bul",
                "Fibonacci serisinin 10. elemanını hesapla",
                "Yarıçapı 5 olan dairenin alanını hesapla"
            ]
            
            for query in test_queries:
                try:
                    logger.info(f"Soru: {query}")
                    result = await llm.generate_str(message=query)
                    logger.info(f"Cevap: {result}")
                    print("-" * 50)
                except Exception as e:
                    logger.error(f"Hata: {e}")
        
        # String Agent'ını test et  
        async with string_agent:
            logger.info("=== STRING AGENT TESTİ ===")
            
            llm = await string_agent.attach_llm(OpenAIAugmentedLLM)
            
            string_queries = [
                "Bu metni tersine çevir: 'Merhaba Dünya'",
                "Bu cümledeki kelime sayısını bul: 'Python ile MCP agent geliştiriyorum'",
                "Bu metni büyük harfe çevir: 'mcp agent çok güçlü'"
            ]
            
            for query in string_queries:
                try:
                    logger.info(f"Soru: {query}")
                    result = await llm.generate_str(message=query)
                    logger.info(f"Cevap: {result}")
                    print("-" * 50)
                except Exception as e:
                    logger.error(f"Hata: {e}")

if __name__ == "__main__":
    print("🚀 Gelişmiş MCP Agent örneği başlatılıyor...")
    start = time.time()
    asyncio.run(advanced_example())
    end = time.time()
    print(f"✅ Toplam çalışma süresi: {end - start:.2f} saniye")
