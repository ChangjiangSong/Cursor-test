"""
军用无人机侦查任务Agent群组系统主入口
"""
import time
from typing import List, Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage

import config
from models import llm
from supervisor.state import SupervisorState, TaskType, TaskStatus
from supervisor.workflow import supervisor_app
from aircraft_simulator import simulator, AircraftStatus

def aircraft_status_callback(old_status, new_status):
    """无人机状态变更回调函数"""
    print(f"\n===> 状态变更通知: {old_status.value} -> {new_status.value}")
    # 这里可以添加针对特定状态变更的处理逻辑

def run_mission(target_area: Dict[str, float]):
    """
    运行无人机侦查任务
    
    Args:
        target_area (Dict[str, float]): 目标区域坐标，包含 north, south, east, west 边界值
    """
    print("初始化任务...")
    
    # 注册无人机状态变更回调
    simulator.register_status_change_callback(aircraft_status_callback)
    
    # 准备初始状态
    initial_state = {
        "messages": [
            SystemMessage(content=config.SUPERVISOR_PROMPT),
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
        "aircraft_status": simulator.current_status.value,
        "next_node": "分析任务"
    }
    
    print("开始任务规划...")
    
    # 启动Supervisor工作流
    result = supervisor_app.invoke(initial_state)
    print("初始规划完成，开始执行任务...")
    
    # 启动任务执行
    simulator.start_mission()
    print("飞机任务模拟已启动，开始状态循环...")
    
    try:
        # 任务执行监控循环
        iteration = 0
        while simulator._running:
            # 每10次迭代打印当前状态
            if iteration % 10 == 0:
                print(f"\n当前循环次数: {iteration}")
                print(f"飞机状态: {simulator.current_status.value}")
                print(f"任务状态: {result['task_status'].value}")
                print(f"当前步骤: {result['current_step']} - {result['execution_plan'][result['current_step']-1].name if result['current_step'] <= len(result['execution_plan']) else '无'}")
                print(f"下一节点: {result['next_node']}")
            
            # 更新当前状态
            current_state = {
                **result,
                "aircraft_status": simulator.current_status.value
            }
            
            # 重新运行工作流
            result = supervisor_app.invoke(current_state)
            
            # 检查任务是否完成
            if result["task_status"] == TaskStatus.COMPLETED:
                print("\n任务已完成！")
                break
                
            # 暂停一段时间再次检查
            time.sleep(1)
            iteration += 1
            
    except KeyboardInterrupt:
        print("\n任务被用户中断")
    finally:
        # 确保停止任务模拟
        simulator.stop_mission()
        print("任务模拟已停止")
    
    # 返回最终结果
    return {
        "task_status": result["task_status"],
        "execution_plan": result["execution_plan"],
        "sar_targets": result["sar_targets"],
        "eo_targets": result["eo_targets"]
    }

def print_mission_summary(result):
    """打印任务摘要"""
    print("\n===== 任务执行摘要 =====")
    print(f"任务状态: {result['task_status'].value}")
    
    print("\n执行步骤:")
    for step in result["execution_plan"]:
        print(f"  - {step.name}: {step.status}")
        
    print("\nSAR目标:")
    for target in result["sar_targets"]:
        print(f"  - ID: {target.get('target_id')}, 置信度: {target.get('confidence'):.2f}, "
              f"坐标: ({target.get('coordinates', {}).get('lat', 0):.4f}, {target.get('coordinates', {}).get('lon', 0):.4f})")
        
    print("\n光电目标:")
    for target in result["eo_targets"]:
        print(f"  - ID: {target.get('target_id')}, 类型: {target.get('target_type')}, 置信度: {target.get('confidence'):.2f}")
        print(f"    坐标: ({target.get('coordinates', {}).get('lat', 0):.4f}, {target.get('coordinates', {}).get('lon', 0):.4f})")
        print(f"    详情: {target.get('details')}")
        
    print("\n===== 任务执行完毕 =====")

if __name__ == "__main__":
    # 设置目标区域
    target_area = {
        "north": 35.18,
        "south": 35.12,
        "east": 117.55,
        "west": 117.45
    }
    
    # 运行任务
    print("\n========= 开始无人机侦查任务 =========")
    result = run_mission(target_area)
    
    # 打印任务摘要
    print_mission_summary(result) 