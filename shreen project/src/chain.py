"""
src/chain.py
─────────────────────────────────────────────────────────────────
Responsibility: Build the full Conversational RAG pipeline.

Components:
  1. Custom empathetic PromptTemplate
  2. ConversationBufferMemory  → tracks chat history
  3. History-aware retriever   → rewrites follow-up questions
                                 into standalone queries before
                                 hitting the vector store
  4. Conversational RAG chain  → retrieval + LLM + memory
─────────────────────────────────────────────────────────────────
"""

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from src.ingest import load_vectorstore, get_retriever

load_dotenv()

# ── LLM Setup ────────────────────────────────────────────────────
# Using ChatGroq (blazing fast, free tier available).
# Swap model_name to any Groq-supported model:
#   "llama3-8b-8192" / "llama3-70b-8192" / "mixtral-8x7b-32768"
# To switch to OpenAI later:
#   from langchain_openai import ChatOpenAI
#   llm = ChatOpenAI(model="gpt-4o", temperature=0)

def get_llm():
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,            # low temp = consistent, factual answers
        max_tokens=350,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )


# ────────────────────────────────────────────────────────────────
# PROMPT 1 — Condense the follow-up question
# ────────────────────────────────────────────────────────────────
# Problem: If a user says "how do I fix that?" after asking about
# a BSOD, the retriever has no idea what "that" refers to.
#
# Solution: We send the chat history + new question to the LLM
# first, asking it to rewrite the question as a fully standalone
# query ("How do I fix a Blue Screen of Death?") BEFORE retrieval.
# ────────────────────────────────────────────────────────────────

CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Given the conversation history and a follow-up question, 
rephrase the follow-up question to be a fully self-contained, 
standalone question that can be understood without the chat history.

Rules:
- Do NOT answer the question.
- Do NOT add information that wasn't in the original question.
- If the question is already standalone, return it unchanged.
- If it's a greeting or small talk, return it exactly as-is."""
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


# ────────────────────────────────────────────────────────────────
# PROMPT 2 — The main empathetic IT Support prompt
# ────────────────────────────────────────────────────────────────
# Design decisions:
#  • Empathetic tone   → users are frustrated; acknowledge first
#  • Numbered steps    → structured, easy to follow
#  • Anti-hallucination guard → only answer from context
#  • Greeting handler  → don't fabricate tech fixes for "hello"
#  • Escalation clause → know when to hand off to human IT
# ────────────────────────────────────────────────────────────────

IT_SUPPORT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are Alex, a warm and conversational IT Support Specialist — 
like a knowledgeable friend who happens to know a lot about computers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONVERSATION STYLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Talk like a real person, not a helpdesk ticket system.
- Use natural transitions between steps like:
  "First...", "Then...", "After that...", "Finally..."
- Add brief human commentary between steps where it helps:
  "This one usually does the trick 👇"
  "Don't worry, this part is easier than it sounds."
  "Almost there —"
- Keep empathy SHORT — one casual sentence max.
  Good: "Ugh, blue screens are the worst — let's sort this out."
  Bad : "I understand how frustrating this must be for you and I want
         to acknowledge your feelings before we begin troubleshooting."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SMALL TALK & GREETINGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Greetings (hi, hello, good morning, hey):
  Respond warmly and naturally. Ask what they need help with.
  NEVER suggest technical fixes for a greeting.
- Small talk (mmh, ok, thanks, cool, lol, 😊):
  Respond like a human would — briefly and warmly.
  Examples:
    "mmh"   → "Hmm, sounds like you're thinking it over 😄 
               What would you like to try next?"
    "ok"    → "Sure thing! Let me know how it goes."
    "thanks" → "Happy to help! Let me know if anything else comes up 😊"
    "lol"   → "Haha 😄 Tech can be wild sometimes. Anything else I can help with?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHEN STEPS DON'T WORK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If the user says the fix didn't work, NEVER immediately give up.
Instead, act like a real technician — dig deeper:

1. First ask a clarifying question to understand what happened:
   "Ah okay — did it fail at a specific step, or did you go through 
    all of them and still see the same issue?"
   "Did anything change at all, even slightly?"
   "Are you getting any error message or error code?"

2. If they give more detail → try a deeper fix from the context.
3. Only suggest the helpdesk after genuinely exhausting options:
   "At this point it might need a hands-on look — 
    your IT helpdesk would be the best next step here."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATTING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- For technical fixes: use a warm intro sentence, then numbered steps.
- Add a natural closing line — vary it, don't always repeat the same phrase.
  Examples:
    "Give those a try and let me know how it goes! 🤞"
    "That should sort it — ping me if anything looks off."
    "Let me know if step 3 trips you up, it can be a bit tricky."
- NEVER use the exact phrase "Let me know if any step needs 
  clarification or if the issue persists" every single time.
  Vary your closing naturally.
- Do NOT invent steps not found in the context.
- Do NOT mix fixes from unrelated issues.
- If context is truly empty and it's a real tech question:
  "Hmm, I don't have that one in my notes right now — 
   but your IT helpdesk should be able to sort it quickly!"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT SOURCE AWARENESS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- If context starts with '[ Answer from IT Knowledge Base ]':
  Answer confidently from the steps provided.
- If context starts with '[ Answer from Web Search ]':
  Summarize naturally into steps. End with:
  "⚠️ Heads up — this is based on a web search, so double-check 
   with your IT team if you're unsure."
- If context is empty and it's a real tech question:
  Ask a follow-up question to better understand the issue
  before saying you don't know.

CONTEXT:
{context}""",
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


# ── In-memory session store ───────────────────────────────────────
# Holds one ChatMessageHistory object per session_id.
# In production, swap this with RedisChatMessageHistory or
# SQLChatMessageHistory for persistence across server restarts.
session_store: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    """Return existing history or create a new one for the session."""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


def format_docs(docs) -> str:
    """
    Flatten retrieved Document objects into a single string block
    that can be injected into the prompt's {context} variable.
    Each doc is separated by a divider for clarity.
    """
    if not docs:
        return "No relevant documents found."
    return "\n\n---\n\n".join([
        f"[{doc.metadata.get('category', 'General')}]\n{doc.page_content}"
        for doc in docs
    ])


# ── Main chain builder ────────────────────────────────────────────

def build_chain():
    """
    Assemble and return the full conversational RAG chain.

    Flow diagram:
    ┌─────────────────────────────────────────────────┐
    │  User Question + Chat History                   │
    │         │                                       │
    │         ▼                                       │
    │  CONDENSE_QUESTION_PROMPT + LLM                 │
    │    → standalone_question                        │
    │         │                                       │
    │         ▼                                       │
    │  Chroma Retriever (k=3)                         │
    │    → top-3 relevant Documents                   │
    │         │                                       │
    │         ▼                                       │
    │  IT_SUPPORT_PROMPT + LLM                        │
    │    → empathetic, numbered answer                │
    │         │                                       │
    │         ▼                                       │
    │  RunnableWithMessageHistory                     │
    │    → auto-saves Q&A to session memory           │
    └─────────────────────────────────────────────────┘
    """

    llm       = get_llm()
    vs        = load_vectorstore()
    retriever = get_retriever(vs, k=5)

    # ── Step 1: Condense follow-up into standalone question ───────
    condense_chain = (
        CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser()
    )

    def condense_question(inputs: dict) -> str:
        """
        If there's no chat history yet (first message), skip the
        LLM call entirely and return the question as-is.
        This saves latency and tokens on the very first turn.
        """
        if not inputs.get("chat_history"):
            return inputs["question"]
        return condense_chain.invoke(inputs)

    # ── Step 2: Retrieve + format docs ───────────────────────────
    def retrieve_and_format(inputs: dict) -> dict:
        """
        Takes the condensed standalone question, retrieves the top-k
        documents from Chroma, formats them, and passes everything
        forward to the final answer prompt.
        """
        standalone_q = condense_question(inputs)
        docs         = retriever.invoke(standalone_q)
        return {
            "context"     : format_docs(docs),
            "question"    : inputs["question"],   # show original Q to LLM
            "chat_history": inputs.get("chat_history", []),
        }

    # ── Step 3: Final answer chain ────────────────────────────────
    answer_chain = (
        RunnableLambda(retrieve_and_format)
        | IT_SUPPORT_PROMPT
        | llm
        | StrOutputParser()
    )

    # ── Step 4: Wrap with message history ─────────────────────────
    # RunnableWithMessageHistory automatically:
    #   • Loads chat history before each invoke
    #   • Appends the new HumanMessage + AIMessage after each invoke
    # You just pass session_id and it handles the rest.
    chain_with_history = RunnableWithMessageHistory(
        answer_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return chain_with_history


# ── Quick CLI test ────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Test the chain end-to-end in the terminal before wiring up the UI.
    Simulates a multi-turn conversation to verify memory works.

        python -m src.chain
    """

    print("=" * 55)
    print("  IT Support Chatbot — Chain Test (multi-turn)")
    print("=" * 55)

    chain = build_chain()

    # Shared config — ties all turns to the same memory session
    config = {"configurable": {"session_id": "test-session-001"}}

    test_turns = [
        "Hello!",
        "My laptop is extremely slow after logging in.",
        "What if the problem is memory related?",   # ← follow-up (tests memory)
        "How do I check that?",                     # ← vague follow-up (tests condensing)
    ]

    for turn in test_turns:
        print(f"\n{'─'*55}")
        print(f"  You : {turn}")
        print(f"{'─'*55}")
        response = chain.invoke(
            {"question": turn},
            config=config,
        )
        print(f"  Alex: {response}")

    print(f"\n{'='*55}")
    print("  Memory stored in session: test-session-001")
    history = get_session_history("test-session-001")
    print(f"  Total messages in history: {len(history.messages)}")
    print("=" * 55)