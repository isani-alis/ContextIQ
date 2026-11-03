from __future__ import annotations

from typing import AsyncIterator, List, TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from backend.retrieval.vector_store import VectorStore
from backend.retrieval.reranker import rerank, is_confident
from backend.ingestion.parser import Chunk

# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------

class AgentState(TypedDict):
    query: str
    condensed_query: str            # rewritten query factoring in history
    history: List[dict]             # [{role, content}] — full conversation
    candidates: List[tuple]         # [(Chunk, float)]
    reranked: List[tuple]           # [(Chunk, float)]
    confident: bool
    answer: str
    citations: List[dict]           # [{source, page, chunk_type, score}]
    context: str                    # built context string, reused for streaming


# ------------------------------------------------------------------
# Node: condense query using conversation history
# ------------------------------------------------------------------

def condense_node(state: AgentState, llm: ChatOllama) -> AgentState:
    """
    If there is prior history, rewrite the current question to be
    self-contained so retrieval doesn't depend on pronouns like
    'he', 'it', 'they', 'the previous one', etc.
    """
    history = state["history"]
    if not history:
        return {**state, "condensed_query": state["query"]}

    # Build a short conversation summary for the rewrite prompt
    turns = []
    for msg in history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        turns.append(f"{role}: {msg['content'][:200]}")
    history_text = "\n".join(turns)

    messages = [
        SystemMessage(content=(
            "Given the conversation history below and a follow-up question, "
            "rewrite the follow-up question to be fully self-contained "
            "so it can be understood without the history. "
            "Output ONLY the rewritten question, nothing else."
        )),
        HumanMessage(content=(
            f"Conversation history:\n{history_text}\n\n"
            f"Follow-up question: {state['query']}\n\n"
            f"Rewritten question:"
        )),
    ]
    response = llm.invoke(messages)
    condensed = response.content.strip().strip('"')
    return {**state, "condensed_query": condensed}


# ------------------------------------------------------------------
# Node: retrieve using condensed query
# ------------------------------------------------------------------

def retrieve_node(state: AgentState, store: VectorStore) -> AgentState:
    candidates = store.hybrid_search(state["condensed_query"], top_k=15)
    return {**state, "candidates": candidates}


# ------------------------------------------------------------------
# Node: rerank
# ------------------------------------------------------------------

def rerank_node(state: AgentState) -> AgentState:
    reranked = rerank(state["condensed_query"], state["candidates"], top_k=5)
    confident = is_confident(reranked)
    return {**state, "reranked": reranked, "confident": confident}


# ------------------------------------------------------------------
# Node: build context + citations (no LLM call — used by both
#       blocking synthesise and streaming endpoints)
# ------------------------------------------------------------------

def prepare_context_node(state: AgentState) -> AgentState:
    if not state["confident"]:
        return {**state, "context": "", "citations": []}

    context_parts = []
    citations = []
    for rank, (chunk, score) in enumerate(state["reranked"]):
        page_label = f"page {chunk.page + 1}" if chunk.page >= 0 else "table"
        context_parts.append(
            f"[{rank + 1}] ({chunk.source}, {page_label})\n{chunk.text}"
        )
        citations.append({
            "source": chunk.source,
            "page": chunk.page + 1 if chunk.page >= 0 else None,
            "chunk_type": chunk.chunk_type,
            "score": round(score, 4),
        })

    return {**state, "context": "\n\n".join(context_parts), "citations": citations}


# ------------------------------------------------------------------
# Node: synthesise (blocking — used by /query)
# ------------------------------------------------------------------

def synthesise_node(state: AgentState, llm: ChatOllama) -> AgentState:
    if not state["confident"]:
        return {
            **state,
            "answer": "I don't have enough context in the uploaded documents to answer this question confidently.",
        }

    messages = _build_messages(state)
    response = llm.invoke(messages)
    return {**state, "answer": response.content}


# ------------------------------------------------------------------
# Streaming synthesise — called directly from API, not a graph node
# ------------------------------------------------------------------

async def stream_answer(state: AgentState, llm: ChatOllama) -> AsyncIterator[str]:
    """Async generator that yields LLM tokens one by one."""
    if not state["confident"]:
        yield "I don't have enough context in the uploaded documents to answer this question confidently."
        return

    messages = _build_messages(state)
    async for chunk in llm.astream(messages):
        if chunk.content:
            yield chunk.content


# ------------------------------------------------------------------
# Shared: build LangChain message list
# ------------------------------------------------------------------

def _build_messages(state: AgentState) -> list:
    messages = [
        SystemMessage(content=(
            "You are ContextIQ, an enterprise document assistant. "
            "Answer the user's question using ONLY the provided context. "
            "Reference sources by their bracketed number [1], [2], etc. "
            "If the context doesn't contain the answer, say so explicitly. "
            "Be concise and precise."
        ))
    ]
    # Inject last 3 turns of conversation for multi-turn awareness
    for msg in state["history"][-6:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=(
        f"Context:\n{state['context']}\n\nQuestion: {state['query']}"
    )))
    return messages


# ------------------------------------------------------------------
# Graph builder
# ------------------------------------------------------------------

def build_graph(store: VectorStore, llm: ChatOllama) -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("condense",         lambda s: condense_node(s, llm))
    graph.add_node("retrieve",         lambda s: retrieve_node(s, store))
    graph.add_node("rerank",           rerank_node)
    graph.add_node("prepare_context",  prepare_context_node)
    graph.add_node("synthesise",       lambda s: synthesise_node(s, llm))

    graph.set_entry_point("condense")
    graph.add_edge("condense",        "retrieve")
    graph.add_edge("retrieve",        "rerank")
    graph.add_edge("rerank",          "prepare_context")
    graph.add_edge("prepare_context", "synthesise")
    graph.add_edge("synthesise",      END)

    return graph.compile()


# ------------------------------------------------------------------
# Partial graph — runs up to prepare_context, returns state
# Used by the streaming endpoint to get citations before streaming
# ------------------------------------------------------------------

def build_retrieval_graph(store: VectorStore, llm: ChatOllama) -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("condense",        lambda s: condense_node(s, llm))
    graph.add_node("retrieve",        lambda s: retrieve_node(s, store))
    graph.add_node("rerank",          rerank_node)
    graph.add_node("prepare_context", prepare_context_node)

    graph.set_entry_point("condense")
    graph.add_edge("condense",        "retrieve")
    graph.add_edge("retrieve",        "rerank")
    graph.add_edge("rerank",          "prepare_context")
    graph.add_edge("prepare_context", END)

    return graph.compile()
