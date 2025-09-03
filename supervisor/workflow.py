"""
Supervisor工作流模块，使用LangGraph构建Agent协作工作流
"""
from typing import Dict, List, Any, Tuple, Literal
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langsmith.run_helpers import traceable
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

from models import llm
from config import SUPERVISOR_PROMPT
from supervisor.state import SupervisorState, TaskType, TaskStatus, TaskStep
from aircraft_simulator import AircraftStatus, simulator
from agents.planning_agents import SARPlanningAgent, EOPlanningAgent, SARPlanningInput, EOPlanningInput
from agents.processing_agents import SARProcessingAgent, EOProcessingAgent, SARProcessingInput, EOProcessingInput

# 创建InMemorySaver实例用于工作流状态保存
checkpointer = InMemorySaver()

# 创建Agent实例
sar_planning_agent = SARPlanningAgent()
eo_planning_agent = EOPlanningAgent()
sar_processing_agent = SARProcessingAgent()
eo_processing_agent = EOProcessingAgent()

# 用于临时存储数据的全局变量
sar_temp_route = None
sar_temp_targets = None
eo_temp_route = None

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
        task="规划目标区域的SAR侦查航线",
        context={
            "target_area": target_area,
            "aircraft_status": state["aircraft_status"]
        }
    )
    
    # 调用SAR规划Agent
    planning_output = sar_planning_agent.process(planning_input)
    
    # 添加规划结果到消息中
    planning_result = f"""
    SAR航线规划结果:
    
    航线描述: {planning_output.result}
    
    航线点:
    {planning_output.route.waypoints if planning_output.route else "无航线点"}
    
    高度: {planning_output.route.altitude if planning_output.route else "未指定"} 米
    速度: {planning_output.route.speed if planning_output.route else "未指定"} 米/秒
    """
    
    messages.append(HumanMessage(content=planning_result))
    
    # 将路由信息保存在全局状态中供后续使用
    global sar_temp_route
    sar_temp_route = planning_output.route.dict() if planning_output.route else None
    
    # 添加人在回路交互：使用interrupt暂停执行并等待人工确认
    interrupt_msg = f"SAR航线规划已完成，请审核以下航线是否满足任务要求:\n\n{planning_result}\n\n请选择: 'accept' 接受航线, 'edit' 修改航线"
    return interrupt(interrupt_msg)
    
    # 以下代码会在用户响应后由main.py中的处理逻辑继续执行
    # 处理人工交互结果将在工作流重新运行时处理

@traceable(run_type="chain")
def _continue_sar_route(state: SupervisorState, response: Dict) -> SupervisorState:
    """继续SAR航线规划流程(在用户交互后)"""
    messages = state["messages"]
    execution_plan = state["execution_plan"]
    current_step = state["current_step"]
    
    # 获取当前步骤
    step = execution_plan[current_step - 1]
    
    # 获取临时存储的航线数据
    global sar_temp_route
    
    # 如果没有临时存储的数据，使用plan_sar_route函数中的规划结果
    if not sar_temp_route:
        # 创建SAR规划输入
        planning_input = SARPlanningInput(
            task="规划目标区域的SAR侦查航线",
            context={
                "target_area": state["target_area"],
                "aircraft_status": state["aircraft_status"]
            }
        )
        
        # 调用SAR规划Agent
        planning_output = sar_planning_agent.process(planning_input)
        
        # 使用规划结果
        if planning_output.route:
            sar_temp_route = planning_output.route.dict()
    
    # 处理人工交互结果
    if response["type"] == "accept":
        # 用户接受航线，继续执行
        messages.append(AIMessage(content="SAR航线已被接受，继续执行任务。"))
    elif response["type"] == "edit":
        # 用户要求修改航线
        # 在实际应用中，这里应该处理用户的修改意见并重新规划航线
        messages.append(HumanMessage(content=f"航线修改请求：{response.get('args', {}).get('feedback', '未提供具体修改意见')}"))
        messages.append(AIMessage(content="已接收航线修改请求，正在重新规划。"))
        
        # 这里可以添加航线修改逻辑，例如：
        # 1. 重新调用SAR规划Agent
        # 2. 或直接修改航线参数
        
        # 简化示例：假设接收了新的航线数据
        if "route_data" in response.get("args", {}):
            sar_temp_route = response["args"]["route_data"]
    else:
        # 未知响应类型
        messages.append(AIMessage(content=f"收到未知响应类型: {response['type']}，默认接受航线继续执行。"))
    
    # 更新步骤状态
    step.status = "已完成"
    step.results = {"route": sar_temp_route if sar_temp_route else {}}
    execution_plan[current_step - 1] = step
    
    # 设置SAR航线到模拟器
    if sar_temp_route:
        simulator.set_sar_route(sar_temp_route)
    
    # 清理临时变量
    route_data = sar_temp_route
    sar_temp_route = None
    
    # 更新状态
    return {
        **state,
        "messages": messages,
        "execution_plan": execution_plan,
        "sar_route": route_data,  # 确保返回SAR航线
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
    
    # 格式化处理结果
    targets_info = "\n".join([
        f"- 目标 {target.target_id}: 置信度 {target.confidence:.2f}, 坐标 ({target.coordinates.lat:.4f}, {target.coordinates.lon:.4f})"
        for target in processing_output.targets
    ])
    
    processing_result = f"""
    SAR数据处理结果:
    
    总结: {processing_output.result}
    
    检测到的目标数量: {len(processing_output.targets)}
    
    目标列表:
    {targets_info}
    """
    
    # 添加处理结果到消息中
    messages.append(HumanMessage(content=processing_result))
    
    # 将目标存储在全局变量中，以便后续处理
    global sar_temp_targets
    sar_temp_targets = [target.dict() for target in processing_output.targets]
    
    # 添加人在回路交互：使用interrupt暂停执行并等待人工确认
    interrupt_msg = f"SAR数据处理已完成，请审核以下检测结果:\n\n{processing_result}\n\n请选择: 'accept' 接受检测结果, 'filter' 过滤目标, 'add' 添加漏检目标"
    
    return interrupt(interrupt_msg)

@traceable(run_type="chain")
def _continue_sar_processing(state: SupervisorState, response: Dict) -> SupervisorState:
    """继续SAR数据处理流程(在用户交互后)"""
    messages = state["messages"]
    execution_plan = state["execution_plan"]
    current_step = state["current_step"]
    
    # 获取当前步骤
    step = execution_plan[current_step - 1]
    
    # 从全局变量中获取临时存储的目标
    global sar_temp_targets
    targets = sar_temp_targets if sar_temp_targets else []
    
    # 处理人工交互结果
    if response["type"] == "accept":
        # 用户接受检测结果，继续执行
        messages.append(AIMessage(content="SAR检测结果已被接受，继续执行任务。"))
    elif response["type"] == "filter":
        # 用户要求过滤目标
        filter_ids = response.get("args", {}).get("filter_ids", [])
        messages.append(HumanMessage(content=f"目标过滤请求：移除目标 {filter_ids}"))
        
        # 过滤目标
        if filter_ids:
            targets = [target for target in targets if target.get("target_id") not in filter_ids]
            messages.append(AIMessage(content=f"已过滤指定目标，剩余 {len(targets)} 个目标。"))
    elif response["type"] == "add":
        # 用户要求添加漏检目标
        new_targets = response.get("args", {}).get("new_targets", [])
        messages.append(HumanMessage(content=f"添加漏检目标请求：添加 {len(new_targets)} 个目标"))
        
        # 添加新目标
        if new_targets:
            # 在实际应用中需要转换为正确的目标对象
            # 这里简化处理
            targets.extend(new_targets)
            messages.append(AIMessage(content=f"已添加指定目标，现有 {len(targets)} 个目标。"))
    else:
        # 未知响应类型
        messages.append(AIMessage(content=f"收到未知响应类型: {response['type']}，默认接受检测结果继续执行。"))
    
    # 更新步骤状态
    step.status = "已完成"
    step.results = {"targets": targets}
    execution_plan[current_step - 1] = step
    
    # 清理临时变量
    targets_data = targets
    sar_temp_targets = None
    
    # 更新状态
    return {
        **state,
        "messages": messages,
        "execution_plan": execution_plan,
        "sar_targets": targets_data,
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
    planning_result = f"""
    光电航线规划结果:
    
    航线描述: {planning_output.result}
    
    目标侦察点:
    {planning_output.route.waypoints if planning_output.route else "无航线点"}
    
    高度: {planning_output.route.altitude if planning_output.route else "未指定"} 米
    速度: {planning_output.route.speed if planning_output.route else "未指定"} 米/秒
    """
    
    messages.append(HumanMessage(content=planning_result))
    
    # 存储路由信息到全局变量
    global eo_temp_route
    eo_temp_route = planning_output.route.dict() if planning_output.route else None
    
    # 添加人在回路交互：使用interrupt暂停执行并等待人工确认
    interrupt_msg = (
        f"光电航线规划已完成，基于SAR探测到的目标生成了光电跟踪航线。请审核以下航线是否满足任务要求:\n\n{planning_result}\n\n"
        f"SAR已识别目标: {len(sar_targets)} 个\n\n"
        f"请选择: 'accept' 接受航线, 'edit' 修改航线, 'prioritize' 调整目标优先级"
    )
    
    return interrupt(interrupt_msg)

@traceable(run_type="chain")
def _continue_eo_route(state: SupervisorState, response: Dict) -> SupervisorState:
    """继续光电航线规划流程(在用户交互后)"""
    messages = state["messages"]
    execution_plan = state["execution_plan"]
    current_step = state["current_step"]
    
    # 获取当前步骤
    step = execution_plan[current_step - 1]
    
    # 获取临时存储的路由
    global eo_temp_route
    eo_route = eo_temp_route
    
    # 处理人工交互结果
    if response["type"] == "accept":
        # 用户接受航线，继续执行
        messages.append(AIMessage(content="光电航线已被接受，继续执行任务。"))
    elif response["type"] == "edit":
        # 用户要求修改航线
        messages.append(HumanMessage(content=f"航线修改请求：{response.get('args', {}).get('feedback', '未提供具体修改意见')}"))
        messages.append(AIMessage(content="已接收光电航线修改请求，正在调整。"))
        
        # 这里可以添加航线修改逻辑
        if "route_data" in response.get("args", {}):
            eo_route = response["args"]["route_data"]
    elif response["type"] == "prioritize":
        # 用户要求调整目标优先级
        priority_targets = response.get("args", {}).get("priority_targets", [])
        messages.append(HumanMessage(content=f"目标优先级调整请求：优先处理目标 {priority_targets}"))
        messages.append(AIMessage(content="已接收目标优先级调整请求，正在重新规划光电航线。"))
        
        # 这里可以添加基于优先级的航线重规划逻辑
        # 简化示例：假设我们记录了优先级信息
        step.results["priority_targets"] = priority_targets
    else:
        # 未知响应类型
        messages.append(AIMessage(content=f"收到未知响应类型: {response['type']}，默认接受航线继续执行。"))
    
    # 更新步骤状态
    step.status = "已完成"
    step.results = {**step.results, "route": eo_route if eo_route else {}}
    execution_plan[current_step - 1] = step
    
    # 设置光电航线到模拟器
    if eo_route:
        simulator.set_eo_route(eo_route)
    
    # 清理临时变量
    route_data = eo_route
    eo_temp_route = None
    
    # 更新状态
    return {
        **state,
        "messages": messages,
        "execution_plan": execution_plan,
        "eo_route": route_data,
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
    # 创建工作流图
    workflow = StateGraph(SupervisorState)
    
    # 添加节点
    workflow.add_node("分析任务", analyze_task)
    workflow.add_node("规划执行", plan_execution)
    workflow.add_node("状态监控", monitor_status)
    workflow.add_node("规划SAR航线", plan_sar_route)
    workflow.add_node("处理SAR数据", process_sar_data)
    workflow.add_node("规划光电航线", plan_eo_route)
    workflow.add_node("处理光电数据", process_eo_data)
    
    # 添加人在回路处理节点
    workflow.add_node("继续SAR航线", _continue_sar_route)
    workflow.add_node("继续SAR处理", _continue_sar_processing)
    workflow.add_node("继续光电航线", _continue_eo_route)
    
    # 设置入口点
    workflow.set_entry_point("分析任务")
    
    # 添加条件边
    for node in ["分析任务", "规划执行", "状态监控", "处理光电数据",
                "继续SAR航线", "继续SAR处理", "继续光电航线"]:
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
    
    # 添加人在回路边缘连接
    workflow.add_edge("规划SAR航线", "继续SAR航线")
    workflow.add_edge("处理SAR数据", "继续SAR处理")
    workflow.add_edge("规划光电航线", "继续光电航线")
    
    return workflow

# 创建supervisor工作流图
supervisor_workflow = build_supervisor_workflow()

# 编译工作流，将checkpointer作为参数传入
supervisor_app = supervisor_workflow.compile(checkpointer=checkpointer) 