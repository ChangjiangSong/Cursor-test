"""
Supervisor工作流模块，使用LangGraph构建Agent协作工作流
"""
from typing import Dict, List, Any, Tuple, Literal
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langsmith.run_helpers import traceable

from models import llm
from config import SUPERVISOR_PROMPT
from supervisor.state import SupervisorState, TaskType, TaskStatus, TaskStep
from aircraft_simulator import AircraftStatus, simulator
from agents.planning_agents import SARPlanningAgent, EOPlanningAgent, SARPlanningInput, EOPlanningInput
from agents.processing_agents import SARProcessingAgent, EOProcessingAgent, SARProcessingInput, EOProcessingInput

# 创建Agent实例
sar_planning_agent = SARPlanningAgent()
eo_planning_agent = EOPlanningAgent()
sar_processing_agent = SARProcessingAgent()
eo_processing_agent = EOProcessingAgent()

@traceable(run_type="llm")
def analyze_task(state: SupervisorState) -> SupervisorState:
    """分析任务，确定任务类型和执行步骤"""
    messages = state["messages"]
    target_area = state["target_area"]
    
    # 构建提示信息
    prompt = f"""
    请分析以下任务信息，确定任务类型并规划执行步骤：
    
    目标区域坐标：
    北边界: {target_area.get('north', 'N/A')}
    南边界: {target_area.get('south', 'N/A')}
    东边界: {target_area.get('east', 'N/A')}
    西边界: {target_area.get('west', 'N/A')}
    
    当前无人机状态: {state['aircraft_status']}
    
    请确定:
    1. 这是什么类型的任务？
    2. 完成这个任务需要哪些步骤？
    3. 每个步骤需要哪些条件才能开始执行？
    """
    
    # 添加提示到消息中
    messages.append(HumanMessage(content=prompt))
    
    # 调用LLM分析任务
    llm_messages = [
        SystemMessage(content=SUPERVISOR_PROMPT),
        *messages
    ]
    response = llm.invoke(llm_messages)
    messages.append(response)
    
    # 这里简化处理，直接设定为区域侦察任务
    # 在实际应用中应该解析LLM响应来获取任务类型和步骤
    
    # 设置任务类型
    task_type = TaskType.AREA_RECONNAISSANCE
    
    # 创建执行计划
    execution_plan = [
        TaskStep(
            step_id=1,
            name="规划SAR航线",
            status="待执行",
            agent="SAR航线规划Agent",
            start_condition="无人机在地面",
            dependent_steps=[]
        ),
        TaskStep(
            step_id=2,
            name="执行SAR任务",
            status="待执行",
            start_condition="无人机到达SAR航线",
            dependent_steps=[1]
        ),
        TaskStep(
            step_id=3,
            name="处理SAR数据",
            status="待执行",
            agent="SAR图像处理Agent",
            start_condition="SAR数据可用",
            dependent_steps=[2]
        ),
        TaskStep(
            step_id=4,
            name="规划光电航线",
            status="待执行",
            agent="光电航线规划Agent",
            start_condition="SAR目标已识别",
            dependent_steps=[3]
        ),
        TaskStep(
            step_id=5,
            name="执行光电任务",
            status="待执行",
            start_condition="无人机到达光电航线",
            dependent_steps=[4]
        ),
        TaskStep(
            step_id=6,
            name="处理光电数据",
            status="待执行",
            agent="光电图像处理Agent",
            start_condition="光电数据可用",
            dependent_steps=[5]
        )
    ]
    
    # 更新状态
    return {
        **state,
        "messages": messages,
        "task_type": task_type,
        "task_status": TaskStatus.PLANNING,
        "execution_plan": execution_plan,
        "current_step": 1,
        "next_node": "规划执行"
    }

@traceable(run_type="chain")
def plan_execution(state: SupervisorState) -> SupervisorState:
    """规划任务执行"""
    messages = state["messages"]
    current_step = state["current_step"]
    execution_plan = state["execution_plan"]
    aircraft_status = state["aircraft_status"]
    
    # 获取当前步骤
    if current_step > len(execution_plan):
        return {**state, "next_node": "任务完成", "task_status": TaskStatus.COMPLETED}
    
    step = execution_plan[current_step - 1]
    
    # 构建提示信息
    prompt = f"""
    当前执行步骤: {step.name} (步骤 {current_step}/{len(execution_plan)})
    步骤状态: {step.status}
    开始条件: {step.start_condition}
    当前无人机状态: {aircraft_status}
    
    请决定下一步操作:
    """
    
    # 添加提示到消息中
    messages.append(HumanMessage(content=prompt))
    
    # 调用LLM决定下一步操作
    llm_messages = [
        SystemMessage(content=SUPERVISOR_PROMPT),
        *messages
    ]
    response = llm.invoke(llm_messages)
    messages.append(response)
    
    # 根据步骤决定下一个节点
    next_node = "状态监控"  # 默认回到状态监控
    
    # 根据当前步骤和无人机状态决定下一个动作
    if step.name == "规划SAR航线" and step.status == "待执行":
        step.status = "执行中"
        next_node = "规划SAR航线"
    elif step.name == "处理SAR数据" and step.status == "待执行" and aircraft_status == AircraftStatus.ON_SAR_MISSION.value:
        step.status = "执行中"
        next_node = "处理SAR数据"
    elif step.name == "规划光电航线" and step.status == "待执行" and len(state["sar_targets"]) > 0:
        step.status = "执行中"
        next_node = "规划光电航线"
    elif step.name == "处理光电数据" and step.status == "待执行" and aircraft_status == AircraftStatus.ON_EO_MISSION.value:
        step.status = "执行中"
        next_node = "处理光电数据"
    
    # 更新步骤状态
    execution_plan[current_step - 1] = step
    
    # 更新状态
    return {
        **state,
        "messages": messages,
        "execution_plan": execution_plan,
        "task_status": TaskStatus.EXECUTING,
        "next_node": next_node
    }

@traceable(run_type="chain")
def monitor_status(state: SupervisorState) -> SupervisorState:
    """监控任务状态"""
    messages = state["messages"]
    execution_plan = state["execution_plan"]
    current_step = state["current_step"]
    aircraft_status = state["aircraft_status"]
    
    # 当前步骤已完成，移动到下一步
    if current_step <= len(execution_plan) and execution_plan[current_step - 1].status == "已完成":
        current_step += 1
    
    # 所有步骤已完成
    if current_step > len(execution_plan):
        prompt = "所有任务步骤已完成，任务执行成功！"
        messages.append(HumanMessage(content=prompt))
        
        llm_messages = [
            SystemMessage(content=SUPERVISOR_PROMPT),
            *messages
        ]
        response = llm.invoke(llm_messages)
        messages.append(response)
        
        return {
            **state,
            "messages": messages,
            "current_step": current_step,
            "task_status": TaskStatus.COMPLETED,
            "next_node": "任务完成"
        }
    
    # 更新当前状态，准备下一步操作
    prompt = f"""
    当前无人机状态: {aircraft_status}
    当前执行步骤: {execution_plan[current_step - 1].name} (步骤 {current_step}/{len(execution_plan)})
    步骤状态: {execution_plan[current_step - 1].status}
    
    请监控任务状态并决定下一步操作:
    """
    
    # 添加提示到消息中
    messages.append(HumanMessage(content=prompt))
    
    # 调用LLM监控状态
    llm_messages = [
        SystemMessage(content=SUPERVISOR_PROMPT),
        *messages
    ]
    response = llm.invoke(llm_messages)
    messages.append(response)
    
    # 更新状态
    return {
        **state,
        "messages": messages,
        "current_step": current_step,
        "next_node": "规划执行"
    }

@traceable(run_type="chain")
def plan_sar_route(state: SupervisorState) -> SupervisorState:
    """规划SAR航线"""
    messages = state["messages"]
    target_area = state["target_area"]
    execution_plan = state["execution_plan"]
    current_step = state["current_step"]
    
    # 获取当前步骤
    step = execution_plan[current_step - 1]
    
    # 创建SAR规划输入
    planning_input = SARPlanningInput(
        task="规划SAR区域侦查航线",
        context={
            "target_area": target_area,
            "aircraft_status": state["aircraft_status"]
        }
    )
    
    # 调用SAR规划Agent
    planning_output = sar_planning_agent.process(planning_input)
    
    # 添加规划结果到消息中
    prompt = f"SAR航线规划结果: {planning_output.result}"
    messages.append(HumanMessage(content=prompt))
    messages.append(AIMessage(content="收到SAR航线规划结果，航线已设置。"))
    
    # 模拟人在回路确认
    prompt = "请确认SAR航线是否满足要求？(是/否)"
    messages.append(HumanMessage(content=prompt))
    messages.append(AIMessage(content="是，航线满足任务要求。"))
    
    # 更新步骤状态
    step.status = "已完成"
    step.results = {"route": planning_output.route.dict()} if planning_output.route else {}
    execution_plan[current_step - 1] = step
    
    # 设置SAR航线到模拟器
    if planning_output.route:
        simulator.set_sar_route(planning_output.route.dict())
    
    # 更新状态
    return {
        **state,
        "messages": messages,
        "execution_plan": execution_plan,
        "sar_route": planning_output.route.dict() if planning_output.route else None,
        "next_node": "状态监控"
    }

@traceable(run_type="chain")
def process_sar_data(state: SupervisorState) -> SupervisorState:
    """处理SAR数据"""
    messages = state["messages"]
    execution_plan = state["execution_plan"]
    current_step = state["current_step"]
    
    # 获取当前步骤
    step = execution_plan[current_step - 1]
    
    # 获取SAR数据
    sar_data = simulator.get_sar_data()
    if not sar_data:
        messages.append(HumanMessage(content="SAR数据暂未获取，等待数据..."))
        return {**state, "messages": messages, "next_node": "状态监控"}
    
    # 创建SAR处理输入
    processing_input = SARProcessingInput(
        task="处理SAR图像数据，识别目标",
        context={"aircraft_status": state["aircraft_status"]},
        sar_data=sar_data
    )
    
    # 调用SAR处理Agent
    processing_output = sar_processing_agent.process(processing_input)
    
    # 添加处理结果到消息中
    prompt = f"SAR数据处理结果: {processing_output.result}"
    messages.append(HumanMessage(content=prompt))
    messages.append(AIMessage(content="SAR数据处理完成，已识别目标。"))
    
    # 更新步骤状态
    step.status = "已完成"
    step.results = {"targets": [target.dict() for target in processing_output.targets]}
    execution_plan[current_step - 1] = step
    
    # 更新状态
    return {
        **state,
        "messages": messages,
        "execution_plan": execution_plan,
        "sar_targets": [target.dict() for target in processing_output.targets],
        "next_node": "状态监控"
    }

@traceable(run_type="chain")
def plan_eo_route(state: SupervisorState) -> SupervisorState:
    """规划光电航线"""
    messages = state["messages"]
    sar_targets = state["sar_targets"]
    execution_plan = state["execution_plan"]
    current_step = state["current_step"]
    
    # 获取当前步骤
    step = execution_plan[current_step - 1]
    
    # 创建光电规划输入
    planning_input = EOPlanningInput(
        task="根据SAR目标规划光电侦查航线",
        context={
            "sar_targets": sar_targets,
            "aircraft_status": state["aircraft_status"]
        }
    )
    
    # 调用光电规划Agent
    planning_output = eo_planning_agent.process(planning_input)
    
    # 添加规划结果到消息中
    prompt = f"光电航线规划结果: {planning_output.result}"
    messages.append(HumanMessage(content=prompt))
    messages.append(AIMessage(content="收到光电航线规划结果，航线已设置。"))
    
    # 模拟人在回路确认
    prompt = "请确认光电航线是否满足要求？(是/否)"
    messages.append(HumanMessage(content=prompt))
    messages.append(AIMessage(content="是，航线满足任务要求。"))
    
    # 更新步骤状态
    step.status = "已完成"
    step.results = {"route": planning_output.route.dict() if planning_output.route else {}}
    execution_plan[current_step - 1] = step
    
    # 设置光电航线到模拟器
    if planning_output.route:
        simulator.set_eo_route(planning_output.route.dict())
    
    # 更新状态
    return {
        **state,
        "messages": messages,
        "execution_plan": execution_plan,
        "eo_route": planning_output.route.dict() if planning_output.route else None,
        "next_node": "状态监控"
    }

@traceable(run_type="chain")
def process_eo_data(state: SupervisorState) -> SupervisorState:
    """处理光电数据"""
    messages = state["messages"]
    execution_plan = state["execution_plan"]
    current_step = state["current_step"]
    
    # 获取当前步骤
    step = execution_plan[current_step - 1]
    
    # 获取光电数据
    eo_data = simulator.get_eo_data()
    if not eo_data:
        messages.append(HumanMessage(content="光电数据暂未获取，等待数据..."))
        return {**state, "messages": messages, "next_node": "状态监控"}
    
    # 创建光电处理输入
    processing_input = EOProcessingInput(
        task="处理光电图像数据，确认目标",
        context={"aircraft_status": state["aircraft_status"]},
        eo_data=eo_data
    )
    
    # 调用光电处理Agent
    processing_output = eo_processing_agent.process(processing_input)
    
    # 添加处理结果到消息中
    prompt = f"光电数据处理结果: {processing_output.result}"
    messages.append(HumanMessage(content=prompt))
    messages.append(AIMessage(content="光电数据处理完成，已确认目标。"))
    
    # 更新步骤状态
    step.status = "已完成"
    step.results = {"targets": [target.dict() for target in processing_output.targets]}
    execution_plan[current_step - 1] = step
    
    # 更新状态
    return {
        **state,
        "messages": messages,
        "execution_plan": execution_plan,
        "eo_targets": [target.dict() for target in processing_output.targets],
        "next_node": "状态监控"
    }

def router(state: SupervisorState) -> Literal[
    "分析任务", "规划执行", "状态监控", "规划SAR航线", "处理SAR数据", "规划光电航线", "处理光电数据", "任务完成"
]:
    """路由函数，决定下一个节点"""
    return state["next_node"]

def build_supervisor_workflow() -> StateGraph:
    """构建Supervisor工作流图"""
    workflow = StateGraph(SupervisorState)
    
    # 添加节点
    workflow.add_node("分析任务", analyze_task)
    workflow.add_node("规划执行", plan_execution)
    workflow.add_node("状态监控", monitor_status)
    workflow.add_node("规划SAR航线", plan_sar_route)
    workflow.add_node("处理SAR数据", process_sar_data)
    workflow.add_node("规划光电航线", plan_eo_route)
    workflow.add_node("处理光电数据", process_eo_data)
    
    # 设置入口点
    workflow.set_entry_point("分析任务")
    
    # 添加条件边
    # 修复错误：不使用空字符串作为源节点，而是将路由器附加到所有节点
    for node in ["分析任务", "规划执行", "状态监控", "规划SAR航线", "处理SAR数据", "规划光电航线", "处理光电数据"]:
        workflow.add_conditional_edges(
            node,
            router,
            {
                "分析任务": "分析任务",
                "规划执行": "规划执行",
                "状态监控": "状态监控",
                "规划SAR航线": "规划SAR航线",
                "处理SAR数据": "处理SAR数据",
                "规划光电航线": "规划光电航线",
                "处理光电数据": "处理光电数据",
                "任务完成": END
            }
        )
    
    return workflow

# 创建supervisor工作流图
supervisor_workflow = build_supervisor_workflow()
supervisor_app = supervisor_workflow.compile() 