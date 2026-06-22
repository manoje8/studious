class RouterAgent:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.config = {
            "enable_confidence_scoring": True,
            "context_window": 5,
            "fallback_category": "factual",
        }

    async def classify(self, question: str, conversation_history: list = None) -> dict:
        context_prompt = self._build_context_prompt(conversation_history)

        prompt = f"""
You are an enterprise-grade intent router for a RAG (Retrieval-Augmented Generation) system.

{context_prompt}

Current question: {question}

CLASSIFICATION TAXONOMY:

1. **FACTUAL** - Direct information retrieval
   - Single fact queries: "What is X?", "When did Y happen?"
   - Entity lookup: "Who is the CEO?", "What's the stock price?"
   - Definition requests: "Define X", "What does X mean?"
   - Attribution: "Who said X?", "Where is X located?"
   - Action: Single-pass retrieval, direct answer
   - Retrieval: 3-5 top chunks, low temperature

2. **COMPARATIVE** - Multi-entity analysis
   - Explicit comparisons: "Compare X vs Y", "Difference between A and B"
   - Decision support: "Which is better?", "What's the best option?"
   - Pros/cons analysis
   - Side-by-side evaluation
   - Action: Multi-query retrieval, structured comparison
   - Retrieval: 5-8 chunks per entity, medium temperature

3. **ANALYTICAL** - Complex reasoning
   - Causal questions: "Why did X happen?", "What caused Y?"
   - Predictive: "What if X?", "How will Y affect Z?"
   - Synthesis: "What are the implications of X?"
   - Root cause analysis
   - Trend analysis: "How has X evolved over time?"
   - Action: Chain-of-thought, multi-step retrieval
   - Retrieval: 8-12 chunks, high temperature, reasoning chains

4. **SUMMARIZATION** - Content condensation
   - Overview: "Summarize X", "Give me an executive summary"
   - Recap: "What did we discuss?", "Summary of previous conversation"
   - Extraction: "Key points from X", "Main takeaways"
   - Document synthesis: "TL;DR of this document"
   - Action: Hierarchical summarization, extractive + abstractive
   - Retrieval: Full document context, low temperature

5. **CHITCHAT** - Social interaction
   - Greetings: "Hello", "Hi", "Good morning"
   - Farewells: "Goodbye", "See you later"
   - Gratitude: "Thank you", "Thanks"
   - Social: "How are you?", "Nice to meet you"
   - Small talk: "How's the weather?", "What's new?"
   - Action: No retrieval, direct response
   - Retrieval: None, use persona-based generation

6. **CLARIFICATION** - Disambiguation requests
   - "What did you mean by X?", "Can you rephrase?"
   - "Tell me more about X", "Elaborate on X"
   - "Could you repeat that?"
   - "I didn't understand X"
   - Action: Targeted follow-up retrieval
   - Retrieval: Context-aware, high precision

7. **PROCEDURAL** - Step-by-step guidance
   - "How do I do X?", "Steps to accomplish Y"
   - "What's the process for Z?", "Guide me through"
   - Tutorials, workflows, recipes
   - Action: Sequential retrieval, structured output
   - Retrieval: Hierarchical, multiple passes

8. **META** - System/context questions
   - "What can you do?", "What are your capabilities?"
   - "Who are you?", "How were you built?"
   - "What do you know about X?"
   - Action: System introspection
   - Retrieval: Minimal, use system prompt

RETRIEVAL STRATEGY INDICATORS:
- needs_hybrid_search: bool (keyword + semantic)
- needs_multi_hop: bool (requires multiple retrieval steps)
- needs_re_ranking: bool (needs additional relevance scoring)
- needs_chunking_strategy: "small"|"medium"|"large"
- confidence_threshold: 0.0-1.0 (minimum confidence required)
- max_retrieval_depth: 1-5 (number of retrieval iterations)

CROSS-CUTTING CONSIDERATIONS:
- Domain specificity (legal, medical, technical, general)
- Urgency/Priority (time-sensitive queries)
- User role/access level implications
- Compliance and regulatory constraints
- Multilingual considerations

Respond with EXACTLY this JSON format:
{{
    "primary_category": "factual|comparative|analytical|summarization|chitchat|clarification|procedural|meta",
    "secondary_categories": ["list", "of", "subcategories"],
    "confidence_score": 0.95,
    "reasoning": "Detailed classification rationale",
    "retrieval_strategy": {{
        "needs_hybrid_search": true,
        "needs_multi_hop": false,
        "needs_re_ranking": true,
        "chunking_strategy": "medium",
        "confidence_threshold": 0.7,
        "max_retrieval_depth": 2,
        "target_chunks": 5
    }},
    "complexity_level": 1-5,
    "estimated_tokens": 500,
    "requires_citation": true,
    "requires_source_attribution": true,
    "suggested_model_temperature": 0.3,
    "suggested_context_window": 4000,
    "priority_level": "normal|high|critical",
    "intent_signals": ["specific", "keywords", "detected"]
}}
"""

        response = await self.llm.complete(prompt)

        result = self._parse_and_validate_response(response)

        return self._apply_fallback_strategy(result)

    def _build_context_prompt(self, conversation_history: list) -> str:
        """Build context-aware prompt from conversation history"""

        if not conversation_history:
            return "No previous conversation context."

        context = "\nPrevious conversation turns:\n"
        for i, turn in enumerate(
            conversation_history[-self.config["context_window"] :], 1
        ):
            context += f"{i}. User: {turn.get('user', '')}\n"
            context += f"   Assistant: {turn.get('assistant', '')}\n"

        context += "\nConversation analysis:\n"
        context += f"- Total turns: {len(conversation_history)}\n"
        context += f"- Turn depth for context: {min(len(conversation_history), self.config['context_window'])}\n"

        if len(conversation_history) > 0:
            context += "- This appears to be a follow-up question that may reference previous context.\n"

        return context

    def _parse_and_validate_response(self, response):
        """Parse and validate the LLM response"""
        try:
            result = response.parsed_json

            required_fields = [
                "primary_category",
                "confidence_score",
                "retrieval_strategy",
            ]
            for field in required_fields:
                if field not in result:
                    result = self._create_fallback_response()
                    break

            if not 0 <= result.get("confidence_score", 0) <= 1:
                result["confidence_score"] = self.config.get("fallback_confidence", 0.7)

            return result

        except (AttributeError, KeyError, TypeError):
            return self._create_fallback_response()

    def _create_fallback_response(self) -> dict:
        """Create a safe fallback response"""

        return {
            "primary_category": self.config["fallback_category"],
            "secondary_categories": [],
            "confidence_score": 0.6,
            "reasoning": "Fallback classification due to parsing error",
            "retrieval_strategy": {
                "needs_hybrid_search": True,
                "needs_multi_hop": False,
                "needs_re_ranking": True,
                "chunking_strategy": "medium",
                "confidence_threshold": 0.5,
                "max_retrieval_depth": 2,
                "target_chunks": 5,
            },
            "complexity_level": 2,
            "estimated_tokens": 500,
            "requires_citation": True,
            "requires_source_attribution": True,
            "suggested_model_temperature": 0.3,
            "suggested_context_window": 4000,
            "priority_level": "normal",
            "intent_signals": ["fallback"],
        }

    def _apply_fallback_strategy(self, result: dict) -> dict:
        """Apply fallback strategies based on confidence and category"""

        # If confidence is low, add uncertainty handling
        if result.get("confidence_score", 0) < 0.7:
            result["requires_confirmation"] = True
            result["fallback_categories"] = ["factual", "analytical"]

        # Ensure procedural questions get high-quality retrieval
        if result["primary_category"] == "procedural":
            result["retrieval_strategy"]["chunking_strategy"] = "large"
            result["retrieval_strategy"]["max_retrieval_depth"] = 3

        # Chitchat should never retrieve
        if result["primary_category"] == "chitchat":
            result["retrieval_strategy"]["target_chunks"] = 0
            result["requires_citation"] = False
            result["requires_source_attribution"] = False

        return result
