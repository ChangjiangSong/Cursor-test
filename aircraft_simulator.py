"""
无人机状态模拟模块，用于模拟飞机的飞行状态和任务执行过程
"""
import time
import threading
from enum import Enum
from typing import Optional, Dict, List, Callable, Any

class AircraftStatus(Enum):
    """无人机状态枚举"""
    ON_GROUND = "地面待命"
    TAKING_OFF = "正在起飞"
    FLYING_TO_SAR = "飞往SAR航线"
    ON_SAR_MISSION = "执行SAR任务中"
    FLYING_TO_EO = "飞往光电航线"
    ON_EO_MISSION = "执行光电任务中"
    RETURNING = "返航中"
    LANDED = "已着陆"

class AircraftSimulator:
    """无人机状态模拟器"""
    
    def __init__(self):
        """初始化模拟器"""
        self.current_status = AircraftStatus.ON_GROUND
        self._status_changed_callbacks: List[Callable[[AircraftStatus, AircraftStatus], Any]] = []
        self._simulation_thread: Optional[threading.Thread] = None
        self._running = False
        self.sar_route: Dict = {}
        self.eo_route: Dict = {}
        self.sar_data: Optional[Dict] = None
        self.eo_data: Optional[Dict] = None
        
        # 配置模拟时间间隔(秒)
        self._time_intervals = {
            "起飞": 1,     # 原为3秒
            "飞行": 2,     # 原为5秒
            "任务": 3,     # 原为8秒
            "返航": 2      # 原为5秒
        }
        
    def register_status_change_callback(self, callback: Callable[[AircraftStatus, AircraftStatus], Any]):
        """注册状态变化回调函数"""
        self._status_changed_callbacks.append(callback)
        
    def _notify_status_change(self, old_status: AircraftStatus, new_status: AircraftStatus):
        """通知所有注册的回调函数状态变化"""
        for callback in self._status_changed_callbacks:
            callback(old_status, new_status)
            
    def set_sar_route(self, route: Dict):
        """设置SAR航线"""
        self.sar_route = route
        print(f"已设置SAR航线: {route}")
        
    def set_eo_route(self, route: Dict):
        """设置光电航线"""
        self.eo_route = route
        print(f"已设置光电航线: {route}")
    
    def get_current_status(self) -> AircraftStatus:
        """获取当前状态"""
        return self.current_status
    
    def get_sar_data(self) -> Optional[Dict]:
        """获取SAR数据"""
        return self.sar_data
    
    def get_eo_data(self) -> Optional[Dict]:
        """获取光电数据"""
        return self.eo_data
        
    def start_mission(self, force=False):
        """
        开始任务模拟
        
        Args:
            force (bool): 当为True时，即使SAR航线未设置也强制启动
        """
        if self._running:
            print("任务模拟已在运行中")
            return
            
        if not self.sar_route and not force:
            print("错误: SAR航线尚未设置，无法开始任务")
            print("提示: 如需强制启动，请使用force=True参数")
            return
            
        if not self.sar_route and force:
            print("警告: SAR航线尚未设置，但以强制模式启动")
            # 设置默认SAR航线
            self.sar_route = {
                "waypoints": "默认航线点",
                "altitude": 3000, 
                "speed": 150,
                "description": "默认生成的SAR航线"
            }
            
        self._running = True
        self._simulation_thread = threading.Thread(target=self._run_simulation)
        self._simulation_thread.daemon = True
        self._simulation_thread.start()
        print("任务模拟已开始")
        
    def stop_mission(self):
        """停止任务模拟"""
        self._running = False
        if self._simulation_thread:
            self._simulation_thread.join(timeout=1.0)
        self.current_status = AircraftStatus.ON_GROUND
        print("任务模拟已停止")
        
    def _run_simulation(self):
        """运行模拟过程"""
        try:
            # 模拟起飞
            print("模拟器: 开始起飞...")
            self._change_status(AircraftStatus.TAKING_OFF)
            time.sleep(self._time_intervals["起飞"])
            
            # 飞往SAR航线
            print("模拟器: 飞往SAR航线...")
            self._change_status(AircraftStatus.FLYING_TO_SAR)
            time.sleep(self._time_intervals["飞行"])
            
            # 执行SAR任务
            print("模拟器: 开始执行SAR任务...")
            self._change_status(AircraftStatus.ON_SAR_MISSION)
            time.sleep(self._time_intervals["任务"])
            
            # 生成模拟SAR数据
            print("模拟器: 生成SAR数据...")
            self.sar_data = {
                "timestamp": time.time(),
                "targets": [
                    {"id": 1, "confidence": 0.87, "coordinates": {"lat": 35.1234, "lon": 117.5678}},
                    {"id": 2, "confidence": 0.76, "coordinates": {"lat": 35.1456, "lon": 117.5912}}
                ],
                "coverage_area": {"north": 35.2, "south": 35.0, "east": 117.7, "west": 117.4}
            }
            
            # 如果没有设置光电航线，则结束任务
            if not self.eo_route and self._running:
                print("模拟器: 未设置光电航线，开始返航...")
                self._change_status(AircraftStatus.RETURNING)
                time.sleep(self._time_intervals["返航"])
                self._change_status(AircraftStatus.LANDED)
                self._running = False
                return
                
            # 飞往光电航线
            if self._running:
                print("模拟器: 飞往光电航线...")
                self._change_status(AircraftStatus.FLYING_TO_EO)
                time.sleep(self._time_intervals["飞行"])
            
            # 执行光电任务
            if self._running:
                print("模拟器: 开始执行光电任务...")
                self._change_status(AircraftStatus.ON_EO_MISSION)
                time.sleep(self._time_intervals["任务"])
                
                # 生成模拟光电数据
                print("模拟器: 生成光电数据...")
                self.eo_data = {
                    "timestamp": time.time(),
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
            
            # 返航
            if self._running:
                print("模拟器: 任务完成，开始返航...")
                self._change_status(AircraftStatus.RETURNING)
                time.sleep(self._time_intervals["返航"])
                
            # 着陆
            if self._running:
                print("模拟器: 着陆...")
                self._change_status(AircraftStatus.LANDED)
                print("模拟器: 任务全部完成")
                
            self._running = False
                
        except Exception as e:
            print(f"模拟过程发生错误: {e}")
            self._running = False
            
    def _change_status(self, new_status: AircraftStatus):
        """改变无人机状态并通知观察者"""
        if not self._running:
            return
            
        old_status = self.current_status
        self.current_status = new_status
        print(f"无人机状态变更: {old_status.value} -> {new_status.value}")
        self._notify_status_change(old_status, new_status)

# 创建全局模拟器实例
simulator = AircraftSimulator() 