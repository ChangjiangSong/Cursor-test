# 军用无人机侦查任务Agent群组系统

基于LangGraph框架构建的军用无人机侦查任务Agent群组系统，实现了supervisor-agent协作框架，用于区域侦查任务的规划与执行。

## 项目结构

```
├── agents/                      # Agent模块
│   ├── __init__.py             
│   ├── common.py                # Agent通用组件
│   ├── planning_agents.py       # 航线规划相关Agent
│   └── processing_agents.py     # 情报处理相关Agent
├── supervisor/                  # Supervisor模块
│   ├── __init__.py
│   ├── state.py                 # 状态定义
│   └── workflow.py              # 工作流定义
├── aircraft_simulator.py        # 无人机状态模拟器
├── config.py                    # 配置文件
├── main.py                      # 主入口
├── models.py                    # 模型服务
└── requirements.txt             # 依赖包
```

## 功能特点

- **统一工作流架构**：使用LangGraph构建分层和顺序执行的工作流
- **模块化设计**：各个组件高内聚低耦合，便于测试和扩展
- **状态监控系统**：实时监控无人机状态和任务执行情况
- **人在回路**：在关键决策点保留人类干预的能力
- **LangSmith集成**：支持全链路追踪和性能监控

## 安装与运行

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行示例

```bash
python main.py
```

## 核心组件

### 1. Supervisor Agent

负责任务分析、规划和协调。对于区域侦查任务，它会：
- 分析任务要求和目标区域信息
- 拆解任务为一系列执行步骤
- 根据无人机状态分配适当的子任务
- 监控任务执行过程

### 2. 航线规划Agent群组

包含两种规划Agent：
- **SAR航线规划Agent**：负责规划SAR载荷的侦查航线
- **光电航线规划Agent**：负责规划光电载荷的侦查航线

### 3. 情报处理Agent群组

包含两种处理Agent：
- **SAR图像处理Agent**：处理SAR图像，识别目标
- **光电图像处理Agent**：处理光电图像，确认目标

### 4. 无人机状态模拟器

模拟无人机的飞行状态和任务执行过程，包括：
- 地面准备、起飞、执行任务、返航等状态
- 模拟SAR和光电数据采集
- 提供状态变更通知机制

## 工作流程

1. 接收区域侦查任务和目标坐标
2. Supervisor分析任务并制定执行计划
3. 调用SAR航线规划Agent规划SAR侦查航线
4. 模拟无人机执行SAR任务
5. 调用SAR图像处理Agent处理数据，识别目标
6. 调用光电航线规划Agent规划光电侦查航线
7. 模拟无人机执行光电任务
8. 调用光电图像处理Agent处理数据，确认目标
9. 生成任务报告和目标信息

## 扩展与优化

本项目设计为原型验证系统，后续可以扩展：

1. 接入真实无人机控制系统
2. 实现真实的图像处理功能
3. 增加更多类型的载荷和Agent
4. 增强决策能力和异常处理
5. 优化Agent间协作机制 