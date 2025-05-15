from typing import Dict, TypedDict, Annotated, List, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_community.chat_models.deepseek import ChatDeepSeek

# 配置DeepSeek模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="sk-f4c8007a2cdd41d796e04fcadb15bb08"
)

# 定义状态类型
class AgentState(TypedDict):
    messages: Annotated[List, "对话历史"]
    thought: Annotated[str, "当前思考过程"]
    next: Annotated[Literal["思考", "回答", "结束"], "下一步操作"]

# 思考节点
def thinking(state: AgentState) -> AgentState:
    """分析用户查询并思考如何回答"""
    # 从状态中获取消息历史
    messages = state["messages"]
    
    # 创建思考提示
    thinking_prompt = [
        SystemMessage(content="分析下面的用户问题，思考如何提供最佳答案。列出你的思考步骤。"),
        messages[-1]  # 用户最后的消息
    ]
    
    # 使用DeepSeek模型生成思考过程
    response = llm.invoke(thinking_prompt)
    
    # 提取思考内容
    thought = response.content
    
    # 更新状态
    return {"messages": messages, "thought": thought, "next": "回答"}

# 回答节点
def answering(state: AgentState) -> AgentState:
    """基于思考过程生成最终回答"""
    # 获取消息历史和思考过程
    messages = state["messages"]
    thought = state["thought"]
    
    # 创建回答提示
    answering_prompt = [
        SystemMessage(content="你是一个有帮助的AI助手。基于以下思考过程，给用户提供简洁清晰的回答："),
        SystemMessage(content=f"思考过程：{thought}"),
        messages[-1]  # 用户最后的消息
    ]
    
    # 使用DeepSeek模型生成回答
    response = llm.invoke(answering_prompt)
    
    # 将AI回答添加到消息历史中
    messages.append(response)
    
    # 更新状态
    return {"messages": messages, "thought": thought, "next": "结束"}

# 定义路由函数
def router(state: AgentState) -> Literal["思考", "回答", "结束"]:
    return state["next"]

# 构建工作流图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("思考", thinking)
workflow.add_node("回答", answering)

# 添加边
workflow.add_edge("思考", "回答")
workflow.add_edge("回答", END)

# 设置入口点
workflow.set_entry_point("思考")

# 设置路由器
workflow.add_conditional_edges("", router, {"思考": "思考", "回答": "回答", "结束": END})

# 编译图
app = workflow.compile()

# 使用示例函数
def run_conversation(query: str, system_prompt: str = "你是一个有帮助的AI助手。"):
    """运行一个高级对话"""
    # 准备初始状态
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]
    
    # 执行图
    result = app.invoke({
        "messages": messages, 
        "thought": "", 
        "next": "思考"
    })
    
    # 返回最后一条消息和思考过程
    return {
        "answer": result["messages"][-1].content,
        "thought": result["thought"]
    }

# 测试示例
if __name__ == "__main__":
    system_prompt = "你是一个专业的AI顾问，提供准确的建议。"
    user_query = "如何有效管理时间？"
    
    response = run_conversation(user_query, system_prompt)
    print(f"问题: {user_query}")
    print(f"思考过程: {response['thought']}")
    print(f"回答: {response['answer']}") 
    