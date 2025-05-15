"""
Agent通用组件和工具函数
"""
from typing import Dict, List, Any, Callable, Optional, TypedDict, Annotated
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel

class AgentInput(BaseModel):
    """Agent输入基类"""
    task: str
    context: Dict[str, Any] = {}
    
class AgentOutput(BaseModel):
    """Agent输出基类"""
    result: str
    status: str = "success"
    details: Dict[str, Any] = {}
    
class AgentHandler:
    """Agent处理器基类"""
    
    def __init__(self, name: str, system_prompt: str, llm: Any):
        """初始化Agent处理器"""
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm
        
    def process(self, input_data: AgentInput) -> AgentOutput:
        """处理输入并返回输出，由子类实现"""
        raise NotImplementedError("子类必须实现process方法")
        
    def _create_messages(self, input_data: AgentInput) -> List:
        """创建消息列表"""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._format_input(input_data))
        ]
        return messages
        
    def _format_input(self, input_data: AgentInput) -> str:
        """格式化输入为字符串"""
        context_str = "\n".join([f"{k}: {v}" for k, v in input_data.context.items()])
        return f"""任务: {input_data.task}\n\n上下文信息:\n{context_str}"""
        
    def _invoke_llm(self, messages: List) -> AIMessage:
        """调用LLM并获取回复"""
        return self.llm.invoke(messages)
        
    def _parse_response(self, response: AIMessage) -> AgentOutput:
        """解析LLM回复，默认实现，可由子类重写"""
        return AgentOutput(
            result=response.content,
            status="success",
            details={}
        ) 