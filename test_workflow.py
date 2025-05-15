"""
Supervisor工作流测试脚本
用于测试LangGraph工作流的各节点功能
"""
from langchain_core.messages import SystemMessage, HumanMessage

from config import SUPERVISOR_PROMPT
from supervisor.state import SupervisorState, TaskType, TaskStatus
from supervisor.workflow import (
    analyze_task, plan_execution, monitor_status, 
    plan_sar_route, process_sar_data, plan_eo_route, process_eo_data
)

def test_analyze_task():
    """测试任务分析节点"""
    print("\n===== 测试任务分析节点 =====")
    
    # 创建初始状态
    target_area = {
        "north": 35.18,
        "south": 35.12,
        "east": 117.55,
        "west": 117.45
    }
    
    initial_state = {
        "messages": [
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(content=f"执行目标区域侦查任务，目标区域边界坐标为：北{target_area['north']}，南{target_area['south']}，东{target_area['east']}，西{target_area['west']}")
        ],
        "task_type": TaskType.UNKNOWN,
        "task_status": TaskStatus.PLANNING,
        "target_area": target_area,
        "execution_plan": [],
        "current_step": 0,
        "sar_route": None,
        "eo_route": None,
        "sar_targets": [],
        "eo_targets": [],
        "aircraft_status": "地面待命",
        "next_node": "分析任务"
    }
    
    # 调用节点函数
    result = analyze_task(initial_state)
    
    # 打印结果
    print(f"任务类型: {result['task_type'].value}")
    print(f"任务状态: {result['task_status'].value}")
    print(f"下一节点: {result['next_node']}")
    print(f"执行计划步骤数: {len(result['execution_plan'])}")
    
    # 打印执行计划
    print("\n执行计划:")
    for step in result["execution_plan"]:
        print(f"  - 步骤{step.step_id}: {step.name}")
        print(f"    状态: {step.status}")
        print(f"    开始条件: {step.start_condition}")
        if step.agent:
            print(f"    执行Agent: {step.agent}")
    
    return result

def test_plan_execution(state):
    """测试任务执行规划节点"""
    print("\n===== 测试任务执行规划节点 =====")
    
    # 调用节点函数
    result = plan_execution(state)
    
    # 打印结果
    print(f"任务状态: {result['task_status'].value}")
    print(f"当前步骤: {result['current_step']}")
    print(f"下一节点: {result['next_node']}")
    
    # 当前步骤状态
    current_step = result["execution_plan"][result["current_step"] - 1]
    print(f"当前步骤: {current_step.name}")
    print(f"步骤状态: {current_step.status}")
    
    return result

def test_workflow_nodes():
    """测试工作流各节点"""
    print("开始工作流节点测试...")
    
    # 测试任务分析节点
    state = test_analyze_task()
    
    # 测试任务执行规划节点
    state = test_plan_execution(state)
    
    # 根据下一个节点继续测试
    if state["next_node"] == "规划SAR航线":
        print("\n===== 测试SAR航线规划节点 =====")
        state = plan_sar_route(state)
        print(f"SAR航线规划完成，下一节点: {state['next_node']}")
        
        # 手动修改状态，模拟无人机已到达SAR航线
        state["aircraft_status"] = "执行SAR任务中"
        
        # 测试状态监控节点
        print("\n===== 测试状态监控节点 =====")
        state = monitor_status(state)
        print(f"状态监控完成，下一节点: {state['next_node']}")
        
        # 测试SAR数据处理节点
        if state["next_node"] == "规划执行":
            # 再次执行规划以进入处理SAR数据节点
            state = plan_execution(state)
            
            if state["next_node"] == "处理SAR数据":
                print("\n===== 测试SAR数据处理节点 =====")
                
                # 模拟设置SAR数据
                from aircraft_simulator import simulator
                simulator.sar_data = {
                    "timestamp": 1686000000,
                    "targets": [
                        {"id": 1, "confidence": 0.87, "coordinates": {"lat": 35.1234, "lon": 117.5678}},
                        {"id": 2, "confidence": 0.76, "coordinates": {"lat": 35.1456, "lon": 117.5912}}
                    ],
                    "coverage_area": {"north": 35.2, "south": 35.0, "east": 117.7, "west": 117.4}
                }
                
                state = process_sar_data(state)
                print(f"SAR数据处理完成，下一节点: {state['next_node']}")
                print(f"识别目标数: {len(state['sar_targets'])}")
    
    print("\n工作流节点测试完成！")

if __name__ == "__main__":
    test_workflow_nodes() 