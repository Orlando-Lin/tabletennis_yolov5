import torch
import cv2
import numpy as np
import os
import sys
import time

# 创建一个模拟 GPIO 类，用于非树莓派环境
class MockGPIO:
    BCM = 'BCM'
    OUT = 'OUT'
    HIGH = 1
    LOW = 0
    
    @staticmethod
    def setmode(mode):
        print(f"模拟 GPIO: 设置模式为 {mode}")
    
    @staticmethod
    def setup(pin, mode):
        print(f"模拟 GPIO: 设置引脚 {pin} 为 {mode}")
    
    @staticmethod
    def output(pin, value):
        print(f"模拟 GPIO: 输出引脚 {pin} 值为 {value}")
    
    @staticmethod
    def cleanup():
        print("模拟 GPIO: 清理资源")

# 根据运行环境选择 GPIO 实现
try:
    import RPi.GPIO as GPIO
    print("使用实际的 GPIO")
except ImportError:
    GPIO = MockGPIO()
    print("使用模拟的 GPIO")

class PingPongDetector:
    def __init__(self, weights_path='yolov5/runs/train/pingpong_model/weights/best.pt'):
        # 检查模型文件是否存在
        if not os.path.exists(weights_path):
            print(f"模型文件不存在: {weights_path}")
            print("请先运行训练脚本 train_model.py")
            raise FileNotFoundError(f"找不到模型文件: {weights_path}")
        
        # 加载模型
        try:
            # 设置超时时间
            import socket
            socket.setdefaulttimeout(30)  # 30秒超时
            
            # 检查并克隆 YOLOv5 仓库
            if not os.path.exists('yolov5'):
                print("未找到本地 YOLOv5，正在克隆...")
                import subprocess
                subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5.git'])
                subprocess.run(['pip', 'install', '-r', 'yolov5/requirements.txt'])
                print("YOLOv5 克隆完成")
            
            # 使用本地模型
            print("使用本 YOLOv5 模型...")
            sys.path.append('yolov5')
            from yolov5.models.experimental import attempt_load
            self.model = attempt_load(weights_path)
            
            # 调整检测参数
            self.model.conf = 0.65  # 进一步提高置信度阈值
            self.model.iou = 0.5   # 提高IOU阈值，减少重复检测
            self.model.classes = [0]  # 只检测乒乓球类别
            self.model.max_det = 5    # 进一步减少最大检测数量
            print("模型加载成功！")
            print(f"检测参数: conf={self.model.conf}, iou={self.model.iou}")
            
            # 移动模型到合适的设备
            if torch.backends.mps.is_available():
                self.model.to('mps')
                print("使用 MPS 加速")
            elif torch.cuda.is_available():
                self.model.to('cuda')
                print("使用 CUDA 加速")
            else:
                self.model.to('cpu')
                print("使用 CPU 运行")
                
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("\n请尝试以下解决方案：")
            print("1. 检查网络连接")
            print("2. 确保已经完成模型训练")
            print("3. 检查权重文件路径是否正确")
            print("4. 手动克隆 YOLOv5 仓库")
            print("   git clone https://github.com/ultralytics/yolov5.git")
            print("   pip install -r yolov5/requirements.txt")
            raise
        
        # 初始化串口
        try:
            import serial
            self.serial_port = serial.Serial(
                port='/dev/ttyAMA0',  # 树莓派的串口设备
                baudrate=115200,      # 提高波特率
                timeout=1,
                write_timeout=1
            )
            print("串口初始化成功")
        except Exception as e:
            print(f"串口初始化失败: {e}")
            self.serial_port = None
        
        # 初始化GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(14, GPIO.OUT)  # TXD
        GPIO.setup(15, GPIO.IN)   # RXD
        print("GPIO初始化完成")
    
    def get_ball_position(self, frame_width, center_x):
        """
        判断乒乓球在画面中的位置
        :param frame_width: 画面宽度
        :param center_x: 球的中心x坐标
        :return: 位置标识 ('L', 'M', 'R') 和位置描述
        """
        # 将画面分成三等份
        left_boundary = frame_width // 3
        right_boundary = (frame_width * 2) // 3
        
        if center_x < left_boundary:
            return 'L', "左侧"
        elif center_x > right_boundary:
            return 'R', "右侧"
        else:
            return 'M', "中间"
    
    def send_position_signal(self, position):
        """
        通过串口发送位置信号
        :param position: 位置标识 ('L', 'M', 'R')
        """
        try:
            if isinstance(GPIO, MockGPIO):
                print(f"模拟发送位置信号: {position}")
                return
            
            if self.serial_port and self.serial_port.is_open:
                # 发送位置信号
                signal = f"{position}\n".encode()  # 添加换行符并编码
                self.serial_port.write(signal)
                self.serial_port.flush()  # 确保数据发送完成
                print(f"发送位置信号: {position}")
            else:
                print("串口未打开")
            
        except Exception as e:
            print(f"发送位置信号时出错: {e}")
    
    def detect(self, frame):
        try:
            # 图像预处理
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 获取原始图像尺寸
            height, width = frame.shape[:2]
            
            # 调整图像大小为型输入尺寸
            input_size = 640
            frame_resized = cv2.resize(frame_rgb, (input_size, input_size))
            
            # 归一化
            frame_float = frame_resized.astype(np.float32) / 255.0
            frame_float = frame_float.transpose(2, 0, 1)
            frame_float = np.expand_dims(frame_float, 0)
            frame_tensor = torch.from_numpy(frame_float)

            # 移动到正确的设备
            if torch.backends.mps.is_available():
                frame_tensor = frame_tensor.to('mps')
            elif torch.cuda.is_available():
                frame_tensor = frame_tensor.to('cuda')
            
            # 进行检测
            with torch.no_grad():
                predictions = self.model(frame_tensor)
                
                if isinstance(predictions, (list, tuple)):
                    predictions = predictions[0]
                
                if torch.is_tensor(predictions):
                    predictions = predictions.detach().cpu().numpy()
                
                if isinstance(predictions, list):
                    predictions = predictions[0]
                
                if len(predictions.shape) > 2:
                    predictions = predictions.squeeze()
            
            # 处理检测结果
            detections = []
            result_frame = frame.copy()
            
            if len(predictions.shape) == 2:
                # 收集所有有效的检测结果
                valid_detections = []
                for i in range(len(predictions)):
                    try:
                        pred = predictions[i]
                        box = pred[:4].astype('float32')
                        conf = float(pred[4])
                        
                        if conf > self.model.conf:
                            # 坐标转换
                            x_center = box[0] * width / input_size
                            y_center = box[1] * height / input_size
                            box_width = box[2] * width / input_size
                            box_height = box[3] * height / input_size
                            
                            valid_detections.append({
                                'x_center': x_center,
                                'y_center': y_center,
                                'width': box_width,
                                'height': box_height,
                                'confidence': conf
                            })
                    except Exception as e:
                        print(f"处理检测框 {i} 时出错: {e}")
                        continue
                
                # 非极大值抑制
                if valid_detections:
                    # 按置信度排序
                    valid_detections.sort(key=lambda x: x['confidence'], reverse=True)
                    
                    # 保留最好的检测结果
                    final_detections = []
                    for det in valid_detections:
                        # 检查是否与已保留的检测结果重叠
                        overlap = False
                        for final_det in final_detections:
                            # 计算中心点距离
                            dx = abs(det['x_center'] - final_det['x_center'])
                            dy = abs(det['y_center'] - final_det['y_center'])
                            dist = (dx*dx + dy*dy) ** 0.5
                            
                            # 如果中心点距离小于平均半���，认为是重叠
                            avg_radius = (det['width'] + det['height'] + 
                                        final_det['width'] + final_det['height']) / 8
                            if dist < avg_radius:
                                overlap = True
                                break
                        
                        if not overlap:
                            final_detections.append(det)
                    
                    # 绘制最终的检测结果
                    for det in final_detections:
                        try:
                            # 计算中心点和半径
                            center_x = int(det['x_center'])
                            center_y = int(det['y_center'])
                            radius = int(min(det['width'], det['height']) / 2)
                            radius = max(5, min(radius, 20))  # 限制半径范围
                            
                            # 添加到检测结果列表
                            detections.append({
                                'center_x': center_x,
                                'center_y': center_y,
                                'radius': radius,
                                'confidence': det['confidence']
                            })
                            
                            # 绘制圆形边框
                            cv2.circle(result_frame, 
                                     (center_x, center_y), 
                                     radius,
                                     (0, 255, 0), 
                                     2)
                            
                            # 画十字中心点
                            cross_size = 2
                            cv2.line(result_frame, 
                                    (center_x - cross_size, center_y),
                                    (center_x + cross_size, center_y),
                                    (0, 0, 255), 1)
                            cv2.line(result_frame,
                                    (center_x, center_y - cross_size),
                                    (center_x, center_y + cross_size),
                                    (0, 0, 255), 1)
                            
                            # 只显示高置信度的标签
                            if det['confidence'] > 0.7:
                                label = f"{det['confidence']:.2f}"
                                cv2.putText(result_frame, label,
                                          (center_x - 15, center_y - radius - 5),
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          0.4,
                                          (0, 255, 0), 1)
                        
                        except Exception as e:
                            print(f"绘制检测框时出错: {e}")
                            continue
            
            # 处理检测结果
            if detections:
                # 获取最高置信度的检测结果
                best_detection = max(detections, key=lambda x: x['confidence'])
                
                # 判断位置并发送信号
                position_code, position_text = self.get_ball_position(
                    frame.shape[1], best_detection['center_x'])
                
                # 发送位置信号
                self.send_position_signal(position_code)
                
                # 在画面上显示位置信息
                position_info = f"Position: {position_text}"
                cv2.putText(result_frame, position_info,
                           (10, 40),  # 在球数量下方显示
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5,
                           (0, 255, 0), 1)
                
                print(f"\r检测到 {len(detections)} 个乒乓球，位置: {position_text}", end='')
            
            return result_frame, detections
            
        except Exception as e:
            print(f"处理帧时出错: {e}")
            import traceback
            traceback.print_exc()
            return frame, []
    
    def __del__(self):
        """
        清理资源
        """
        try:
            if hasattr(self, 'serial_port') and self.serial_port:
                self.serial_port.close()
                print("串口已关闭")
            GPIO.cleanup()
            print("GPIO资源已清理")
        except:
            pass

def main():
    try:
        # 初始化检测器
        print("正在加载模型...")
        detector = PingPongDetector()
        
        # 打开摄像头
        print("正在打开摄像头...")
        cap = cv2.VideoCapture(0)
        
        # 检查摄像头是否成功打开
        if not cap.isOpened():
            if sys.platform == "darwin":  # macOS
                print("\n错误: 无法访问摄像头")
                print("请确保已授予摄像头访问权限：")
                print("1. 打开系统偏好设置")
                print("2. 点击'安全性与隐私'")
                print("3. 选择'摄像头'")
                print("4. 确保您的 Python/终端程序已被允许访问摄像头")
            raise Exception("无法打开摄像头")
        
        # 等待摄像头初始化
        print("等待摄像头初始化...")
        time.sleep(2)
        
        # 尝试读取几帧，确保摄像头正常工作
        for _ in range(5):
            ret, frame = cap.read()
            if ret:
                break
            time.sleep(0.5)
        
        if not ret:
            raise Exception("无法从摄像头读取图像")
        
        # 优化相机设置
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)  # 设置帧率
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲
        
        print("摄像头已就绪")
        print(f"分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"帧率: {int(cap.get(cv2.CAP_PROP_FPS))}")
        print("按'q'退出程序")
        
        # 用于计算FPS
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0
        
        while True:
            # 计算FPS
            fps_frame_count += 1
            if fps_frame_count >= 30:  # 每30帧更新一次FPS
                fps = fps_frame_count / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0
            
            ret, frame = cap.read()
            if not ret:
                print("读取摄像头画面失败，尝试重新连接...")
                time.sleep(0.1)  # 减少等待时间
                continue
            
            try:
                # 进行检测
                result_frame, detections = detector.detect(frame)
                
                # 显示FPS
                cv2.putText(result_frame, f"FPS: {fps:.1f}",
                           (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 1)
                
                # 显示结果
                cv2.imshow('PingPong Detection', result_frame)
                
            except Exception as e:
                print(f"\n处理帧时出错: {e}")
                continue
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        print("\n正在清理资源...")
        try:
            cap.release()
            cv2.destroyAllWindows()
            GPIO.cleanup()
        except:
            pass
        print("程序结束")

if __name__ == "__main__":
    main() 