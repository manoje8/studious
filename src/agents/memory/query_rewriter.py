import logfire

from src.agents.memory.conversation_model import ConversationSession


class QueryRewriter:
    def __init__(self, llm_client):
        self.llm = llm_client

    async def rewrite(self, current_message: str, session: ConversationSession) -> dict:
        if not session or not session.turns:
            return {
                "rewritten_query": current_message,
                "was_rewritten": False,
                "resolved_references": [],
            }

        history_text = session.to_prompt_format(n=6)

        prompt = f"""
                You are resolving references in a user's question using conversation history.

                Conversation so far:
                {history_text}

                New user message: {current_message}

                Task: Rewrite the user message as a fully self-contained question that
                can be understood without any conversation history. Resolve all pronouns
                (it, that, they, them, those), references (the report, the same one,
                the previous answer), and implicit context.

                If the message is already self-contained, return it unchanged.

                Respond in JSON only:
                {{
                    "rewritten_query": "the fully self-contained question",
                    "was_rewritten": true/false,
                    "resolved_references": ["list of what you resolved", "e.g. that → Q3 memory usage"]
                }}
                """

        response = await self.llm.complete(prompt)
        result = response.parsed_json

        if result["was_rewritten"]:
            logfire.info(f"Query rewritten: '{current_message}'-> '{result['rewritten_query']}'")

            for ref in result["resolved_references"]:
                logfire.info(f" Resolved: {ref}")

        return result
