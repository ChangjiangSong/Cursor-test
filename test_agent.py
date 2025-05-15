"""
Agent单元测试脚本
用于测试各个独立Agent的功能
"""
import json
from typing import Dict, Any

from agents.planning_agents import SARPlanningAgent, EOPlanningAgent, SARPlanningInput, EOPlanningInput
from agents.processing_agents import SARProcessingAgent, EOProcessingAgent, SARProcessingInput, EOProcessingInput

def test_sar_planning_agent():
    """测试SAR航线规划Agent"""
    print("\n===== 测试SAR航线规划Agent =====")
    
    # 创建测试输入
    target_area = {
        "north": 35.18,
        "south": 35.12,
        "east": 117.55,
        "west": 117.45
    }
    
    input_data = SARPlanningInput(
        task="规划SAR区域侦查航线",
        context={
            "target_area": target_area,
            "aircraft_status": "地面待命"
        }
    )
    
    # 创建Agent并处理
    agent = SARPlanningAgent()
    output = agent.process(input_data)
    
    # 打印结果
    print(f"结果: {output.result}")
    print(f"状态: {output.status}")
    if output.route:
        print(f"航线类型: {output.route.route_type}")
        print(f"航线点数: {len(output.route.waypoints)}")
        print(f"高度: {output.route.altitude}米")
        print(f"飞行速度: {output.route.speed}km/h")
        print(f"预计时间: {output.route.estimated_time}分钟")
    
    return output

def test_eo_planning_agent(sar_targets):
    """测试光电航线规划Agent"""
    print("\n===== 测试光电航线规划Agent =====")
    
    # 创建测试输入
    input_data = EOPlanningInput(
        task="根据SAR目标规划光电侦查航线",
        context={
            "sar_targets": sar_targets,
            "aircraft_status": "执行SAR任务中"
        }
    )
    
    # 创建Agent并处理
    agent = EOPlanningAgent()
    output = agent.process(input_data)
    
    # 打印结果
    print(f"结果: {output.result}")
    print(f"状态: {output.status}")
    if output.route:
        print(f"航线类型: {output.route.route_type}")
        print(f"航线点数: {len(output.route.waypoints)}")
        print(f"高度: {output.route.altitude}米")
        print(f"飞行速度: {output.route.speed}km/h")
        print(f"预计时间: {output.route.estimated_time}分钟")
    
    return output

def test_sar_processing_agent():
    """测试SAR图像处理Agent"""
    print("\n===== 测试SAR图像处理Agent =====")
    
    # 创建模拟SAR数据
    sar_data = {
        "timestamp": 1686000000,
        "targets": [
            {"id": 1, "confidence": 0.87, "coordinates": {"lat": 35.1234, "lon": 117.5678}},
            {"id": 2, "confidence": 0.76, "coordinates": {"lat": 35.1456, "lon": 117.5912}}
        ],
        "coverage_area": {"north": 35.2, "south": 35.0, "east": 117.7, "west": 117.4}
    }
    
    # 创建测试输入
    input_data = SARProcessingInput(
        task="处理SAR图像数据，识别目标",
        context={"aircraft_status": "执行SAR任务中"},
        sar_data=sar_data
    )
    
    # 创建Agent并处理
    agent = SARProcessingAgent()
    output = agent.process(input_data)
    
    # 打印结果
    print(f"结果: {output.result}")
    print(f"状态: {output.status}")
    print(f"识别目标数: {len(output.targets)}")
    for target in output.targets:
        print(f"  - ID: {target.target_id}, 置信度: {target.confidence:.2f}")
        print(f"    坐标: ({target.coordinates['lat']:.4f}, {target.coordinates['lon']:.4f})")
    
    return output

def test_eo_processing_agent():
    """测试光电图像处理Agent"""
    print("\n===== 测试光电图像处理Agent =====")
    
    # 创建模拟光电数据
    eo_data = {
        "timestamp": 1686000000,
        "targets": [
            {
                "id": 1, 
                "type": "vehicle", 
                "confidence": 0.95,
                "coordinates": {"lat": 35.1236, "lon": 117.5680},
                "details": "军用装甲车，伪装网覆盖"
            }
        ],
        "image_quality": "high"
    }
    
    # 创建测试输入
    input_data = EOProcessingInput(
        task="处理光电图像数据，确认目标",
        context={"aircraft_status": "执行光电任务中"},
        eo_data=eo_data
    )
    
    # 创建Agent并处理
    agent = EOProcessingAgent()
    output = agent.process(input_data)
    
    # 打印结果
    print(f"结果: {output.result}")
    print(f"状态: {output.status}")
    print(f"确认目标数: {len(output.targets)}")
    for target in output.targets:
        print(f"  - ID: {target.target_id}, 类型: {target.target_type}, 置信度: {target.confidence:.2f}")
        print(f"    坐标: ({target.coordinates['lat']:.4f}, {target.coordinates['lon']:.4f})")
        print(f"    详情: {target.details}")
    
    return output

def run_agent_tests():
    """运行所有Agent测试"""
    print("开始Agent单元测试...")
    
    # 测试SAR规划Agent
    sar_planning_output = test_sar_planning_agent()
    
    # 测试SAR处理Agent
    sar_processing_output = test_sar_processing_agent()
    sar_targets = [target.dict() for target in sar_processing_output.targets]
    
    # 测试光电规划Agent
    eo_planning_output = test_eo_planning_agent(sar_targets)
    
    # 测试光电处理Agent
    eo_processing_output = test_eo_processing_agent()
    
    print("\n所有Agent测试完成！")

if __name__ == "__main__":
    run_agent_tests() 