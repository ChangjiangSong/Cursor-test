"""
军用无人机侦查任务Agent群组系统主入口
"""
import time
import uuid
from typing import List, Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Command

import config
from models import llm
from supervisor.state import SupervisorState, TaskType, TaskStatus
from supervisor.workflow import supervisor_app
from aircraft_simulator import simulator, AircraftStatus

def aircraft_status_callback(old_status, new_status):
    """无人机状态变更回调函数"""
    print(f"\n===> 状态变更通知: {old_status.value} -> {new_status.value}")
    # 这里可以添加针对特定状态变更的处理逻辑

def process_human_interaction(interrupt_message, task_context=None):
    """
    处理人在回路交互
    
    Args:
        interrupt_message: 中断消息
        task_context: 任务上下文信息，可以包含当前状态等
        
    Returns:
        Command对象，包含人工决策结果
    """
    print("\n" + "="*50)
    print("需要人工干预:")
    print(interrupt_message)
    print("="*50)
    
    # 显示可选操作
    if "accept" in interrupt_message:
        print("\n可选操作:")
        if "规划" in interrupt_message and "SAR" in interrupt_message:
            print("1. accept - 接受SAR航线")
            print("2. edit - 修改SAR航线 (输入修改意见)")
        elif "规划" in interrupt_message and "光电" in interrupt_message:
            print("1. accept - 接受光电航线")
            print("2. edit - 修改光电航线 (输入修改意见)")
            print("3. prioritize - 调整目标优先级")
        elif "SAR数据处理" in interrupt_message:
            print("1. accept - 接受检测结果")
            print("2. filter - 过滤目标 (输入要过滤的目标ID)")
            print("3. add - 添加漏检目标")
    
    # 获取人工输入
    while True:
        try:
            action = input("\n请输入操作 (例如 'accept'): ").strip()
            
            # 处理不同类型的操作
            if action == "accept":
                return Command(resume={"type": "accept"})
            
            elif action == "edit":
                feedback = input("请输入修改意见: ").strip()
                # 在实际系统中，可以支持更复杂的航线编辑，这里简化为文本反馈
                return Command(resume={"type": "edit", "args": {"feedback": feedback}})
            
            elif action == "prioritize" and "光电" in interrupt_message:
                priority_input = input("请输入优先处理的目标ID (多个ID用逗号分隔): ").strip()
                priority_targets = [id.strip() for id in priority_input.split(",")]
                return Command(resume={"type": "prioritize", "args": {"priority_targets": priority_targets}})
            
            elif action == "filter" and "SAR数据处理" in interrupt_message:
                filter_input = input("请输入要过滤的目标ID (多个ID用逗号分隔): ").strip()
                filter_ids = [id.strip() for id in filter_input.split(",")]
                return Command(resume={"type": "filter", "args": {"filter_ids": filter_ids}})
            
            elif action == "add" and "SAR数据处理" in interrupt_message:
                print("添加目标功能需要更复杂的接口，此示例中简化处理")
                target_count = int(input("请输入要添加的目标数量: ").strip())
                new_targets = []
                # 在实际系统中，这里应该是一个更详细的目标添加界面
                print("此示例中将生成空目标占位符")
                return Command(resume={"type": "add", "args": {"new_targets": [{}] * target_count}})
            
            else:
                print("无效的操作，请重新输入")
        
        except Exception as e:
            print(f"输入处理错误: {e}")
            print("请重新输入")

def run_mission(target_area: Dict[str, float]):
    """
    运行无人机侦查任务
    
    Args:
        target_area (Dict[str, float]): 目标区域坐标，包含 north, south, east, west 边界值
    """
    # 初始化结果变量，防止UnboundLocalError
    result = None
    
    try:
        print("初始化任务...")
        
        # 注册无人机状态变更回调
        simulator.register_status_change_callback(aircraft_status_callback)
        
        # 生成会话ID，用于状态保存
        session_id = str(uuid.uuid4())
        print(f"会话ID: {session_id}")
        
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
        
        # 配置checkpointer的必要参数
        config_dict = {
            "configurable": {
                "thread_id": session_id,  # 为checkpointer提供线程ID
                "checkpoint_ns": "uav_mission"  # 命名空间
            }
        }
        
        print("开始任务规划...")
        
        # 首先执行工作流，确保SAR航线已设置
        result = supervisor_app.invoke(initial_state, config=config_dict)
        print("初始规划完成...")
        
        # 检查是否需要人工交互
        if hasattr(result, 'interrupt'):
            print("需要人工交互来设置SAR航线...")
            interrupt_message = result.interrupt
            
            # 获取人工决策
            command = process_human_interaction(interrupt_message)
            
            # 恢复工作流执行
            if hasattr(result, 'state'):
                result = supervisor_app.invoke(command, state=result.state, config=config_dict)
            else:
                result = supervisor_app.invoke(command, config=config_dict)
            
            # 如果结果是字典且包含SAR航线，则将其设置到模拟器
            if isinstance(result, dict) and result.get("sar_route"):
                simulator.set_sar_route(result["sar_route"])
                print("已将SAR航线设置到模拟器")
        
        # 现在启动任务执行
        print("开始任务执行...")
        simulator.start_mission(force=True)  # 使用强制模式启动
        print("飞机任务模拟已启动，开始工作流执行...")
        
        # 任务执行监控循环
        iteration = 0
        while simulator._running:
            # 每10次迭代打印当前状态
            if iteration % 10 == 0 and isinstance(result, dict) and not hasattr(result, 'interrupt'):
                # 只在正常状态下打印状态信息
                try:
                    print(f"\n当前循环次数: {iteration}")
                    print(f"飞机状态: {simulator.current_status.value}")
                    print(f"任务状态: {result['task_status'].value}")
                    print(f"当前步骤: {result['current_step']} - {result['execution_plan'][result['current_step']-1].name if result['current_step'] <= len(result['execution_plan']) else '无'}")
                    print(f"下一节点: {result['next_node']}")
                except (KeyError, TypeError):
                    print("无法显示当前状态信息")
            
            # 检查工作流执行结果
            if hasattr(result, 'interrupt'):
                # 人在回路交互
                interrupt_message = result.interrupt
                
                # 获取人工决策
                command = process_human_interaction(interrupt_message)
                
                # 将状态传递回工作流
                if hasattr(result, 'state'):
                    result = supervisor_app.invoke(command, state=result.state, config=config_dict)
                else:
                    result = supervisor_app.invoke(command, config=config_dict)
            elif isinstance(result, dict):
                # 正常执行流程
                
                # 检查任务是否完成
                if result.get("task_status") == TaskStatus.COMPLETED:
                    print("\n任务已完成！")
                    break
                
                # 更新当前状态
                current_state = {
                    **result,
                    "aircraft_status": simulator.current_status.value
                }
                
                # 重新运行工作流
                result = supervisor_app.invoke(current_state, config=config_dict)
            else:
                # 未知结果类型
                print(f"未知结果类型: {type(result)}")
                break
            
            # 暂停一段时间再次检查
            time.sleep(1)
            iteration += 1
            
    except KeyboardInterrupt:
        print("\n任务被用户中断")
    except Exception as e:
        print(f"\n执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保停止任务模拟
        simulator.stop_mission()
        print("任务模拟已停止")
    
    # 返回最终结果
    if isinstance(result, dict):
        return {
            "task_status": result.get("task_status"),
            "execution_plan": result.get("execution_plan", []),
            "sar_targets": result.get("sar_targets", []),
            "eo_targets": result.get("eo_targets", [])
        }
    else:
        print("任务未正常完成，无法获取结果")
        return None

def print_mission_summary(result):
    """打印任务摘要"""
    print("\n===== 任务执行摘要 =====")
    
    if not result:
        print("任务未正常完成，无法显示摘要信息")
        print("\n===== 任务执行结束 =====")
        return
        
    print(f"任务状态: {result['task_status'].value}")
    
    print("\n执行步骤:")
    for step in result["execution_plan"]:
        print(f"  - {step.name}: {step.status}")
        
    print("\nSAR目标:")
    for target in result["sar_targets"]:
        print(f"  - ID: {target.get('target_id')}, 置信度: {target.get('confidence'):.2f}, "
              f"坐标: ({target.get('coordinates', {}).get('lat', 0):.4f}, {target.get('coordinates', {}).get('lon', 0):.4f})")
        
    print("\n光电目标:")
    for target in result.get("eo_targets", []):
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