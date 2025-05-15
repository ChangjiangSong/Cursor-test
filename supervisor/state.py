"""
Supervisor状态模块，定义系统工作流状态
"""
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Literal
from enum import Enum
from pydantic import BaseModel, Field

from agents.planning_agents import RouteData
from agents.processing_agents import TargetData

class TaskType(str, Enum):
    """任务类型枚举"""
    AREA_RECONNAISSANCE = "区域侦察任务"
    TARGET_TRACKING = "目标跟踪任务"
    UNKNOWN = "未知任务"
    
class TaskStatus(str, Enum):
    """任务状态枚举"""
    PLANNING = "规划中"
    EXECUTING = "执行中"
    COMPLETED = "已完成"
    FAILED = "失败"
    
class TaskStep(BaseModel):
    """任务步骤"""
    step_id: int
    name: str
    status: str = "待执行"
    agent: Optional[str] = None
    start_condition: Optional[str] = None
    dependent_steps: List[int] = []
    results: Dict[str, Any] = {}

class SupervisorState(TypedDict):
    """Supervisor状态"""
    # 消息历史
    messages: Annotated[List, "消息历史"]
    
    # 任务信息
    task_type: Annotated[TaskType, "任务类型"]
    task_status: Annotated[TaskStatus, "任务状态"]
    target_area: Annotated[Dict[str, float], "目标区域坐标"]
    
    # 计划与执行
    execution_plan: Annotated[List[TaskStep], "执行计划步骤"]
    current_step: Annotated[int, "当前执行步骤"]
    
    # 航线数据
    sar_route: Annotated[Optional[Dict], "SAR航线"]
    eo_route: Annotated[Optional[Dict], "光电航线"]
    
    # 目标数据
    sar_targets: Annotated[List[Dict], "SAR目标数据"]
    eo_targets: Annotated[List[Dict], "光电目标数据"]
    
    # 无人机状态
    aircraft_status: Annotated[str, "无人机状态"]
    
    # 工作流控制
    next_node: Annotated[Literal[
        "分析任务", 
        "规划执行", 
        "状态监控", 
        "规划SAR航线", 
        "处理SAR数据", 
        "规划光电航线", 
        "处理光电数据", 
        "任务完成"
    ], "下一个执行节点"] 