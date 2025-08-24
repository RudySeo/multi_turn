# step1_simple_multi_turn.py
from typing import List
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI


# 1) 상태 정의 111
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# 2) LLM 클라이언트
llm = ChatOpenAI(temperature=0)


# 3) AI 답변 노드 (멀티턴 유지)
def answer(state: ChatState) -> ChatState:
    """
    LLM 호출 후 AI 답변을 상태에 누적
    - state['messages']를 그대로 유지
    - 새로운 AIMessage를 추가
    """
    msgs = state["messages"]  # 현재까지의 모든 메시지
    out = llm.invoke(msgs)  # LLM 호출
    return {"messages": msgs + [AIMessage(content=str(out.content).strip())]}


# 4) 그래프 구성
graph = StateGraph(ChatState)
graph.add_node("answer", answer)
graph.add_edge("answer", END)
graph.set_entry_point("answer")
graph = graph.compile()
