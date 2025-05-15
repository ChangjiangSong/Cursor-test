"""
航线规划相关Agent模块
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from langchain_core.messages import AIMessage

from config import SAR_PLANNING_PROMPT, EO_PLANNING_PROMPT
from models import llm
from agents.common import AgentHandler, AgentInput, AgentOutput

class RouteData(BaseModel):
    """航线数据模型"""
    route_type: str
    waypoints: List[Dict[str, float]]
    altitude: float
    speed: float
    estimated_time: float
    coverage_area: Dict[str, float] = {}
    
class SARPlanningInput(AgentInput):
    """SAR规划输入"""
    pass
    
class SARPlanningOutput(AgentOutput):
    """SAR规划输出"""
    route: Optional[RouteData] = None
    
class SARPlanningAgent(AgentHandler):
    """SAR航线规划Agent"""
    
    def __init__(self):
        """初始化SAR规划Agent"""
        super().__init__("SAR航线规划Agent", SAR_PLANNING_PROMPT, llm)
        
    def process(self, input_data: SARPlanningInput) -> SARPlanningOutput:
        """处理SAR规划请求"""
        messages = self._create_messages(input_data)
        response = self._invoke_llm(messages)
        return self._parse_response(response)
        
    def _parse_response(self, response: AIMessage) -> SARPlanningOutput:
        """解析SAR规划响应"""
        # 在实际应用中，这里应该解析详细的航线数据
        # 简化版只返回一个模拟航线
        
        route = RouteData(
            route_type="SAR",
            waypoints=[
                {"lat": 35.12, "lon": 117.45},
                {"lat": 35.12, "lon": 117.55},
                {"lat": 35.18, "lon": 117.55},
                {"lat": 35.18, "lon": 117.45},
                {"lat": 35.12, "lon": 117.45}
            ],
            altitude=5000.0,
            speed=150.0,
            estimated_time=25.0,
            coverage_area={
                "north": 35.18,
                "south": 35.12,
                "east": 117.55,
                "west": 117.45
            }
        )
        
        return SARPlanningOutput(
            result=response.content,
            status="success",
            details={"route_info": "S型扫描航线"},
            route=route
        )

class EOPlanningInput(AgentInput):
    """光电规划输入"""
    pass
    
class EOPlanningOutput(AgentOutput):
    """光电规划输出"""
    route: Optional[RouteData] = None
    
class EOPlanningAgent(AgentHandler):
    """光电航线规划Agent"""
    
    def __init__(self):
        """初始化光电规划Agent"""
        super().__init__("光电航线规划Agent", EO_PLANNING_PROMPT, llm)
        
    def process(self, input_data: EOPlanningInput) -> EOPlanningOutput:
        """处理光电规划请求"""
        messages = self._create_messages(input_data)
        response = self._invoke_llm(messages)
        return self._parse_response(response)
        
    def _parse_response(self, response: AIMessage) -> EOPlanningOutput:
        """解析光电规划响应"""
        # 在实际应用中，这里应该解析详细的航线数据
        # 简化版只返回一个模拟航线
        
        route = RouteData(
            route_type="EO",
            waypoints=[
                {"lat": 35.123, "lon": 117.567},
                {"lat": 35.125, "lon": 117.569},
                {"lat": 35.122, "lon": 117.571},
                {"lat": 35.120, "lon": 117.568},
                {"lat": 35.123, "lon": 117.567}
            ],
            altitude=3000.0,
            speed=120.0,
            estimated_time=15.0
        )
        
        return EOPlanningOutput(
            result=response.content,
            status="success",
            details={"route_info": "环形盘旋观测"},
            route=route
        ) 