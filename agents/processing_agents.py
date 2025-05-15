"""
情报处理相关Agent模块
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from langchain_core.messages import AIMessage

from config import SAR_PROCESSING_PROMPT, EO_PROCESSING_PROMPT
from models import llm
from agents.common import AgentHandler, AgentInput, AgentOutput

class TargetData(BaseModel):
    """目标数据模型"""
    target_id: int
    confidence: float
    coordinates: Dict[str, float]
    target_type: Optional[str] = None
    details: Optional[str] = None
    
class SARProcessingInput(AgentInput):
    """SAR处理输入"""
    sar_data: Dict[str, Any]
    
class SARProcessingOutput(AgentOutput):
    """SAR处理输出"""
    targets: List[TargetData] = []
    
class SARProcessingAgent(AgentHandler):
    """SAR图像处理Agent"""
    
    def __init__(self):
        """初始化SAR处理Agent"""
        super().__init__("SAR图像处理Agent", SAR_PROCESSING_PROMPT, llm)
        
    def process(self, input_data: SARProcessingInput) -> SARProcessingOutput:
        """处理SAR图像数据"""
        messages = self._create_messages(input_data)
        response = self._invoke_llm(messages)
        return self._parse_response(response, input_data)
        
    def _parse_response(self, response: AIMessage, input_data: SARProcessingInput) -> SARProcessingOutput:
        """解析SAR处理响应"""
        # 从输入的SAR数据中提取目标信息
        targets = []
        
        if "targets" in input_data.sar_data:
            for target in input_data.sar_data["targets"]:
                targets.append(TargetData(
                    target_id=target.get("id", 0),
                    confidence=target.get("confidence", 0.0),
                    coordinates=target.get("coordinates", {"lat": 0.0, "lon": 0.0}),
                    target_type="未知",
                    details="SAR图像目标"
                ))
        
        return SARProcessingOutput(
            result=response.content,
            status="success",
            details={"processing_info": "SAR图像已处理并识别目标"},
            targets=targets
        )

class EOProcessingInput(AgentInput):
    """光电处理输入"""
    eo_data: Dict[str, Any]
    
class EOProcessingOutput(AgentOutput):
    """光电处理输出"""
    targets: List[TargetData] = []
    
class EOProcessingAgent(AgentHandler):
    """光电图像处理Agent"""
    
    def __init__(self):
        """初始化光电处理Agent"""
        super().__init__("光电图像处理Agent", EO_PROCESSING_PROMPT, llm)
        
    def process(self, input_data: EOProcessingInput) -> EOProcessingOutput:
        """处理光电图像数据"""
        messages = self._create_messages(input_data)
        response = self._invoke_llm(messages)
        return self._parse_response(response, input_data)
        
    def _parse_response(self, response: AIMessage, input_data: EOProcessingInput) -> EOProcessingOutput:
        """解析光电处理响应"""
        # 从输入的光电数据中提取目标信息
        targets = []
        
        if "targets" in input_data.eo_data:
            for target in input_data.eo_data["targets"]:
                targets.append(TargetData(
                    target_id=target.get("id", 0),
                    confidence=target.get("confidence", 0.0),
                    coordinates=target.get("coordinates", {"lat": 0.0, "lon": 0.0}),
                    target_type=target.get("type", "未知"),
                    details=target.get("details", "光电图像目标")
                ))
        
        return EOProcessingOutput(
            result=response.content,
            status="success",
            details={"processing_info": "光电图像已处理并确认目标"},
            targets=targets
        ) 