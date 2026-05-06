# ================================================
# STRATEGIC HEALTH NAVIGATOR v20.5 
# Flexible Initial + Better Flow Control
# ================================================

from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import json
from dotenv import load_dotenv
import time
load_dotenv()

# ================== LLM ==================
llm = ChatOllama(
    model="phi4-mini",      #qwen3:8b, phi4-mini
    # model="qwen3:8b",      #qwen3:8b, phi4-mini
    temperature=0.65,
)

json_llm = llm.bind(response_format={"type": "json_object"})

class State(TypedDict):
    messages: Annotated[list[BaseMessage], "add"]
    known_info: dict
    current_topic: str
    current_strategy: str
    emotional_tone: str
    visited_topics: list

# ================== PROMPTS ==================

INITIAL_PROMPT = ChatPromptTemplate.from_template("""
You are Maya, a warm and caring health navigator / care buddy.

Conversation history summary: {history_summary}

Generate a natural, friendly opening message (1-2 sentences maximum).

Guidelines:
- If this is the first conversation (no real history), warmly introduce yourself and gently invite them to share how they've been or what's on their mind.
- If this is an ongoing conversation, reference the past lightly and check in on how they are doing today.
- Be warm, human, and purposeful. Avoid empty generic lines like "hope you're feeling great".

Return ONLY the greeting message.
""")
# ================== ANALYZER PROMPT (Defined here) ==================
ANALYZER_PROMPT = ChatPromptTemplate.from_template("""
You are an expert analyst.
Patient said: "{user_message}"
Known information: {known_info}
Current topic: {current_topic}

Return ONLY valid JSON with updates.
{{
  "known_info_update": {{...}},
  "emotional_tone": "positive/neutral/negative/distressed/cooperative/vague/avoidant",
  "topic_completion": {{...}}
}}
""")

PLANNER_PROMPT = ChatPromptTemplate.from_template("""
You are Maya, a warm and strategic health navigator.
You are responsible for gently leading the conversation...

Natural topic flow:
1. Initial rapport & check-in
2. Symptoms & patterns
3. Daily life / functional impact
4. Medication use & adherence
5. Barriers & challenges
6. Coping strategies & support
7. Emotional wellbeing

Current known: {known_info}
Recent summary: {history_summary}
Patient just said: "{user_message}"
Current topic: {current_topic}
Tone: {tone}
Topic completion: {topic_completion}
Visited topics: {visited_topics}

Rules:
- Stay on the current topic unless the patient has shared enough information or clearly wants to change.
- Do not jump topics quickly after 1-2 short replies.
- If the patient says something vague or off-topic, acknowledge it first then gently return to current topic.
- Only progress when it feels natural.

Return ONLY valid JSON:
{{
  "next_topic": "short name of next focus",
  "next_move": "one short sentence - your immediate goal",
  "rationale": "one short sentence explaining your decision",
  "should_progress": true/false
}}
""")

RESPONDER_PROMPT = ChatPromptTemplate.from_template("""
You are Maya, a kind and natural health navigator.

Known: {known_info}
Current focus: {current_topic}
Plan: {next_move}

Patient said: {user_message}

Respond as Maya:
- Warmly acknowledge what they just said (even if short, vague or off-topic)
- Keep tone caring and natural
- Be concise (2-4 sentences max)
- Gently guide according to the current plan
- Always end with one easy, natural question related to the current topic

Maya's reply:
""")

# ================== HELPERS ==================
def get_history_summary(messages):
    if len(messages) < 3:
        return "This is the first conversation."
    recent = messages[-15:]
    text = "\n".join([f"{'Patient' if isinstance(m, HumanMessage) else 'Maya'}: {m.content[:200]}" for m in recent])
    try:
        summary = llm.invoke(f"Summarize the key points of this health conversation in 2-3 sentences:\n{text}").content.strip()
        return summary
    except:
        return "Ongoing health conversation."


def initial_greeting(history_summary: str = "This is the first conversation."):
    try:
        prompt = INITIAL_PROMPT.invoke({"history_summary": history_summary})
        greeting = llm.invoke(prompt).content.strip()
        # Clean possible quotes
        greeting = greeting.strip('"').strip("'")
        return greeting
    except:
        return "Hello! I'm Maya, your care buddy. I'm here to support you. How have you been feeling lately?"


# ================== NODES ==================
def analyzer(state: State):
    user_msg = state["messages"][-1].content
    current_topic = state.get("current_topic", "Initial rapport & check-in")

    prompt = ANALYZER_PROMPT.invoke({
        "user_message": user_msg,
        "known_info": json.dumps(state.get("known_info", {})),
        "current_topic": current_topic
    })
    
    try:
        res = json_llm.invoke(prompt)
        data = json.loads(res.content)
    except:
        data = {"known_info_update": {}, "emotional_tone": "neutral", "topic_progress": 50, "topic_completion": {}}
    
    new_known = {**state.get("known_info", {}), **data.get("known_info_update", {})}
    new_known.setdefault("topic_completion", {}).update(data.get("topic_completion", {}))

    return {
        "known_info": new_known,
        "emotional_tone": data.get("emotional_tone", "neutral")
    }


def planner(state: State):
    user_msg = state["messages"][-1].content
    tone = state.get("emotional_tone", "neutral")
    history = get_history_summary(state["messages"])
    current_topic = state.get("current_topic", "Initial rapport & check-in")
    topic_comp = json.dumps(state.get("known_info", {}).get("topic_completion", {}))
    visited = state.get("visited_topics", [])

    prompt = PLANNER_PROMPT.invoke({
        "known_info": json.dumps(state.get("known_info", {})),
        "history_summary": history,
        "user_message": user_msg,
        "current_topic": current_topic,
        "tone": tone,
        "topic_completion": topic_comp,
        "visited_topics": visited
    })
   
    try:
        res = json_llm.invoke(prompt)
        data = json.loads(res.content)
        
        next_topic = data.get("next_topic", current_topic)
        next_move = data.get("next_move", "Continue warmly")
        rationale = data.get("rationale", "")
        should_progress = data.get("should_progress", False)

    except Exception as e:
        print(f"[Planner Error] {e}")
        next_topic = current_topic
        next_move = "Acknowledge and ask a gentle follow-up"
        rationale = "Error fallback"

    print(f"[DEBUG] Topic: {next_topic} | Progress: {should_progress} | Rationale: {rationale}")

    new_visited = visited + [next_topic] if next_topic not in visited else visited

    return {
        "current_topic": next_topic,
        "current_strategy": next_move,
        "visited_topics": new_visited
    }


def responder(state: State):
    prompt = RESPONDER_PROMPT.invoke({
        "known_info": json.dumps(state.get("known_info", {})),
        "current_topic": state.get("current_topic", ""),
        "next_move": state.get("current_strategy", ""),
        "user_message": state["messages"][-1].content
    })
   
    reply = llm.invoke(prompt).content.strip()
    return {"messages": [AIMessage(content=reply)]}




# ================== GRAPH ==================
def build_navigator():
    graph = StateGraph(State)
    graph.add_node("analyzer", analyzer)
    graph.add_node("planner", planner)
    graph.add_node("responder", responder)
    
    graph.add_edge(START, "analyzer")
    graph.add_edge("analyzer", "planner")
    graph.add_edge("planner", "responder")
    graph.add_edge("responder", END)
    
    return graph.compile(checkpointer=MemorySaver())


# ================== RUN ==================
if __name__ == "__main__":
    print("=== Strategic Health Navigator v20.5 ===\n")
    
    graph = build_navigator()
    config = {"configurable": {"thread_id": "maya_v20_5"}}
    
    # Get initial history summary
    # initial_history = "This is the first conversation."    
    initial_history = ""
    initial = initial_greeting(initial_history)
    print(f"Maya: {initial}\n")
    
    state = {
        "messages": [AIMessage(content=initial)],
        "known_info": {"topic_completion": {}},
        "current_topic": "Initial rapport & check-in",
        "emotional_tone": "neutral",
        "visited_topics": []
    }
    
    while True:
        user_input = input("Patient: ").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Maya: Take care. I'm here whenever you need.")
            break
        if not user_input:
            continue
        start_time = time.time()
        result = graph.invoke({
            "messages": [HumanMessage(content=user_input)],
            "known_info": state.get("known_info", {}),
            "current_topic": state.get("current_topic"),
            "emotional_tone": state.get("emotional_tone"),
            "visited_topics": state.get("visited_topics", [])
        }, config=config)
        
        # Convert to M:SS format
        duration = time.time() - start_time
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        time_str = f"{minutes}:{seconds:02d}"        
        print(f"[{time_str}] Maya: {result['messages'][-1].content}\n")
        
        # Update state
        state["messages"].extend(result.get("messages", []))
        state["known_info"] = result.get("known_info", state["known_info"])
        state["current_topic"] = result.get("current_topic", state["current_topic"])
        state["emotional_tone"] = result.get("emotional_tone", state["emotional_tone"])
        state["visited_topics"] = result.get("visited_topics", state.get("visited_topics", []))