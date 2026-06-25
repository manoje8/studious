import asyncio

from src.llm.groq import GroqClient


async def safe_llm():
    gemini = GroqClient(timeout_seconds=30, max_retries=2)

    response = await gemini.complete('Return JSON: {"key": "value"}', max_tokens=100)

    if response.has_json():
        data = response.parsed_json
        print(f"Parsed: {data}")

    else:
        print("Error in pared json")
        data = response.try_parsed_json(default={"error": "parse_failed"})

    print(f"Total calls: {gemini.total_calls}")
    print(f"Total tokens: {gemini.total_tokens}")


if __name__ == "__main__":
    asyncio.run(safe_llm())
