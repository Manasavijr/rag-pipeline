"""
LLM Chain — LangChain + Ollama for Q&A, summarization, and classification.
All inference is local — no external API keys required.
"""

import logging
from typing import Dict, List, Optional, Tuple

from langchain.schema import Document
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from app.core.config import settings

logger = logging.getLogger(__name__)


# ── Prompt Templates ──────────────────────────────────────────────────────────

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert document analyst. Use ONLY the provided context to answer the question.
If the answer is not in the context, say "I could not find relevant information in the provided documents."

Context:
{context}

Question: {question}

Answer (be specific and cite relevant details from the context):"""
)

SUMMARIZE_PROMPT = PromptTemplate(
    input_variables=["context", "style", "focus"],
    template="""You are an expert at summarizing documents. Summarize the following content.

Style: {style}
Focus: {focus}

Content:
{context}

Summary:"""
)

CLASSIFY_PROMPT = PromptTemplate(
    input_variables=["text", "categories"],
    template="""Classify the following text into exactly ONE of these categories: {categories}

Text:
{text}

Respond in this exact format:
CATEGORY: <category name>
CONFIDENCE: <high/medium/low>
REASONING: <one sentence explanation>"""
)


# ── LLM Chain Manager ────────────────────────────────────────────────────────

class LLMChainManager:
    """
    Manages LangChain + Ollama chains for all RAG operations.
    Connects to locally running Ollama — no API keys needed.
    """

    def __init__(self):
        self._llm: Optional[OllamaLLM] = None
        self._connected: bool = False

    def _get_llm(self, temperature: float = None) -> OllamaLLM:
        """Get or create Ollama LLM instance."""
        temp = temperature if temperature is not None else settings.OLLAMA_TEMPERATURE
        return OllamaLLM(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=temp,
            timeout=settings.OLLAMA_TIMEOUT,
        )

    def check_connection(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            import httpx
            response = httpx.get(
                f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5
            )
            if response.status_code == 200:
                models = [m["name"] for m in response.json().get("models", [])]
                model_base = settings.OLLAMA_MODEL.split(":")[0]
                self._connected = any(model_base in m for m in models)
                if not self._connected:
                    logger.warning(
                        f"Model '{settings.OLLAMA_MODEL}' not found in Ollama. "
                        f"Available: {models}. Run: ollama pull {settings.OLLAMA_MODEL}"
                    )
                return self._connected
        except Exception as e:
            logger.warning(f"Ollama not reachable at {settings.OLLAMA_BASE_URL}: {e}")
            self._connected = False
        return False

    def answer(
        self,
        question: str,
        context_docs: List[Tuple[Document, float]],
        temperature: float = 0.1,
    ) -> str:
        """Run Q&A chain over retrieved documents."""
        if not context_docs:
            return "No relevant documents found to answer your question."

        context = self._format_context(context_docs)
        llm = self._get_llm(temperature)
        chain = LLMChain(llm=llm, prompt=QA_PROMPT)

        logger.info(f"Running Q&A chain with {len(context_docs)} context docs")
        result = chain.invoke({"context": context, "question": question})
        return result["text"].strip()

    def summarize(
        self,
        context_docs: List[Tuple[Document, float]],
        style: str = "concise",
        focus: str = "main topics and key findings",
    ) -> str:
        """Summarize retrieved document chunks."""
        if not context_docs:
            return "No documents found to summarize."

        context = self._format_context(context_docs)
        style_instructions = {
            "concise": "Write a concise 2-3 paragraph summary",
            "detailed": "Write a comprehensive detailed summary covering all major points",
            "bullet": "Write a bullet-point summary with clear categories",
        }
        style_prompt = style_instructions.get(style, style_instructions["concise"])

        llm = self._get_llm(temperature=0.3)
        chain = LLMChain(llm=llm, prompt=SUMMARIZE_PROMPT)

        logger.info(f"Running summarization chain (style={style})")
        result = chain.invoke({
            "context": context,
            "style": style_prompt,
            "focus": focus,
        })
        return result["text"].strip()

    def classify(
        self,
        text: str,
        categories: List[str],
    ) -> Dict[str, str]:
        """
        Classify text into one of the provided categories.
        Returns {category, confidence, reasoning}.
        """
        categories_str = ", ".join(categories)
        llm = self._get_llm(temperature=0.0)
        chain = LLMChain(llm=llm, prompt=CLASSIFY_PROMPT)

        logger.info(f"Running classification chain (categories={categories})")
        result = chain.invoke({"text": text[:3000], "categories": categories_str})
        return self._parse_classification(result["text"], categories)

    def _format_context(self, docs: List[Tuple[Document, float]]) -> str:
        """Format retrieved documents into context string."""
        parts = []
        for i, (doc, score) in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "")
            page_str = f" (page {page})" if page else ""
            parts.append(
                f"[Document {i} — {source}{page_str}, relevance: {score:.2f}]\n"
                f"{doc.page_content}"
            )
        return "\n\n---\n\n".join(parts)

    def _parse_classification(
        self, response: str, categories: List[str]
    ) -> Dict[str, str]:
        """Parse the structured classification response."""
        result = {
            "category": "general",
            "confidence": "low",
            "reasoning": response,
            "all_scores": {cat: "low" for cat in categories},
        }
        lines = response.strip().split("\n")
        for line in lines:
            if line.startswith("CATEGORY:"):
                cat = line.replace("CATEGORY:", "").strip().lower()
                # Find closest matching category
                for c in categories:
                    if c.lower() in cat or cat in c.lower():
                        result["category"] = c
                        result["all_scores"][c] = result.get("confidence", "medium")
                        break
            elif line.startswith("CONFIDENCE:"):
                conf = line.replace("CONFIDENCE:", "").strip().lower()
                result["confidence"] = conf if conf in ("high", "medium", "low") else "medium"
            elif line.startswith("REASONING:"):
                result["reasoning"] = line.replace("REASONING:", "").strip()
        return result

    @property
    def is_connected(self) -> bool:
        return self._connected
