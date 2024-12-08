import torch
import cv2
import numpy as np
import os
import sys
import time

class PingPongDetector:
    def __init__(self, weights_path='yolov5/runs/train/pingpong_model/weights/best.pt'):
        # 检查模型文件是否存在
        if not os.path.exists(weights_path):
            print(f"模型文件不存在: {weights_path}")
            print("请先运行训练脚本 train_model.py")
            raise FileNotFoundError(f"找不到模型文件: {weights_path}")
        
        # 加载模型
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                      path=weights_path, force_reload=True,
                                      trust_repo=True)
            # 调整检测参数
            self.model.conf = 0.3  # 降低置信度阈值，提高检测灵敏度
            self.model.iou = 0.4   # 调整IOU阈值
            self.model.classes = [0]  # 只检测乒乓球类别
            self.model.max_det = 10   # 最多检测10个目标
            print("模型加载成功！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def detect(self, frame):
        # 进行检测
        results = self.model(frame)
        
        # 获取检测结果
        detections = results.pandas().xyxy[0]
        
        # 在图像上绘制检测结果
        result_frame = frame.copy()
        for idx, detection in detections.iterrows():
            x1, y1, x2, y2 = map(int, detection[['xmin', 'ymin', 'xmax', 'ymax']])
            conf = detection['confidence']
            
            # 计算中心点和半径
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius = int(max(x2 - x1, y2 - y1) / 2)
            
            # 画圆形边界
            cv2.circle(result_frame, (center_x, center_y), radius, (0, 255, 0), 2)
            
            # 画十字中心点
            cross_size = 5
            cv2.line(result_frame, 
                    (center_x - cross_size, center_y),
                    (center_x + cross_size, center_y),
                    (0, 0, 255), 2)
            cv2.line(result_frame,
                    (center_x, center_y - cross_size),
                    (center_x, center_y + cross_size),
                    (0, 0, 255), 2)
            
            # 显示位置信息和置信度
            info_text = f"Ball ({center_x},{center_y}) Conf:{conf:.2f}"
            cv2.putText(result_frame, info_text,
                       (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 0), 2)
            
            # 添加半透明高亮效果
            overlay = result_frame.copy()
            cv2.circle(overlay, (center_x, center_y), radius, (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.2, result_frame, 0.8, 0, result_frame)
        
        # 添加检测统计信息
        if len(detections) > 0:
            stats_text = f"Detected: {len(detections)} balls"
            cv2.putText(result_frame, stats_text,
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 255, 0), 2)
        
        return result_frame, detections

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
        
        print("摄像头已就绪")
        print("按'q'键退出程序")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("读取摄像头画面失败，尝试重新连接...")
                time.sleep(0.5)
                continue
            
            try:
                # 调整图像大小以提高性能
                frame = cv2.resize(frame, (640, 480))
                
                # 进行检测
                result_frame, detections = detector.detect(frame)
                
                # 显示结果
                cv2.imshow('PingPong Detection', result_frame)
                
                # 显示检测到的乒乓球数量
                if len(detections) > 0:
                    print(f"\r检测到 {len(detections)} 个乒乓球，置信度: {detections['confidence'].mean():.2f}", end='')
                else:
                    print("\r未检测到乒乓球", end='')
                
            except Exception as e:
                print(f"\n处理帧时出错: {e}")
                continue
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("请确保已经完成模型训练，并且权重文件存在。")
    except Exception as e:
        print(f"\n发生错误: {e}")
    finally:
        print("\n正在清理资源...")
        try:
            cap.release()
            cv2.destroyAllWindows()
        except:
            pass
        print("程序结束")

if __name__ == "__main__":
    main() 