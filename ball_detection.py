import cv2
import numpy as np

def nothing(x):
    pass

def detect_pingpong_ball():
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    # 创建一个窗口
    cv2.namedWindow('HSV Tuner', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('HSV Tuner', 600, 400)  # 设置窗口大小
    
    # 为白色乒乓球创建滑动条，设置初始值
    cv2.createTrackbar('White H_min', 'HSV Tuner', 0, 180, nothing)
    cv2.createTrackbar('White H_max', 'HSV Tuner', 180, 180, nothing)
    cv2.createTrackbar('White S_min', 'HSV Tuner', 0, 255, nothing)
    cv2.createTrackbar('White S_max', 'HSV Tuner', 80, 255, nothing)
    cv2.createTrackbar('White V_min', 'HSV Tuner', 180, 255, nothing)
    cv2.createTrackbar('White V_max', 'HSV Tuner', 255, 255, nothing)
    
    # 为橙色乒乓球创建滑动条，设置初始值
    cv2.createTrackbar('Orange H_min', 'HSV Tuner', 0, 180, nothing)
    cv2.createTrackbar('Orange H_max', 'HSV Tuner', 30, 180, nothing)
    cv2.createTrackbar('Orange S_min', 'HSV Tuner', 100, 255, nothing)
    cv2.createTrackbar('Orange S_max', 'HSV Tuner', 255, 255, nothing)
    cv2.createTrackbar('Orange V_min', 'HSV Tuner', 100, 255, nothing)
    cv2.createTrackbar('Orange V_max', 'HSV Tuner', 255, 255, nothing)

    print("调节器已创建，请调节HSV值以检测乒乓球")
    print("按'q'键退出程序")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        try:
            # 获取滑动条的值
            # 白色范围
            w_h_min = cv2.getTrackbarPos('White H_min', 'HSV Tuner')
            w_h_max = cv2.getTrackbarPos('White H_max', 'HSV Tuner')
            w_s_min = cv2.getTrackbarPos('White S_min', 'HSV Tuner')
            w_s_max = cv2.getTrackbarPos('White S_max', 'HSV Tuner')
            w_v_min = cv2.getTrackbarPos('White V_min', 'HSV Tuner')
            w_v_max = cv2.getTrackbarPos('White V_max', 'HSV Tuner')
            
            # 橙色范围
            o_h_min = cv2.getTrackbarPos('Orange H_min', 'HSV Tuner')
            o_h_max = cv2.getTrackbarPos('Orange H_max', 'HSV Tuner')
            o_s_min = cv2.getTrackbarPos('Orange S_min', 'HSV Tuner')
            o_s_max = cv2.getTrackbarPos('Orange S_max', 'HSV Tuner')
            o_v_min = cv2.getTrackbarPos('Orange V_min', 'HSV Tuner')
            o_v_max = cv2.getTrackbarPos('Orange V_max', 'HSV Tuner')
            
            # 打印当前值，方便调试
            if cv2.waitKey(1) & 0xFF == ord('p'):
                print(f"\n当前HSV值：")
                print(f"白色 - H:{w_h_min}-{w_h_max}, S:{w_s_min}-{w_s_max}, V:{w_v_min}-{w_v_max}")
                print(f"橙色 - H:{o_h_min}-{o_h_max}, S:{o_s_min}-{o_s_max}, V:{o_v_min}-{o_v_max}")
            
            # 设置HSV范围
            lower_white = np.array([w_h_min, w_s_min, w_v_min])
            upper_white = np.array([w_h_max, w_s_max, w_v_max])
            lower_orange = np.array([o_h_min, o_s_min, o_v_min])
            upper_orange = np.array([o_h_max, o_s_max, o_v_max])
            
            # 创建掩码
            mask_white = cv2.inRange(hsv, lower_white, upper_white)
            mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
            mask = cv2.bitwise_or(mask_white, mask_orange)
            
            # 边缘检测
            edges = cv2.Canny(frame, 50, 150)
            
            # 结合边缘和颜色掩码
            mask = cv2.bitwise_and(mask, edges)
            
            # 进行形态学操作
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            
            # 使用霍夫圆变换检测圆
            circles = cv2.HoughCircles(
                mask,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=50,
                param1=50,
                param2=30,
                minRadius=10,
                maxRadius=100
            )
            
            # 显示所有调试图像
            cv2.imshow('Original', frame)
            cv2.imshow('Edges', edges)
            cv2.imshow('Mask', mask)
            
            # 在原始图像上绘制检测结果
            result = frame.copy()
            
            # 如果找到了圆，处理每个圆
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    center = (i[0], i[1])
                    radius = i[2]
                    
                    # 检查圆内的颜色一致性
                    # 创建圆形掩码
                    circle_mask = np.zeros_like(mask)
                    cv2.circle(circle_mask, center, radius, 255, -1)
                    
                    # 计算圆内的掩码像素比例
                    circle_region = cv2.bitwise_and(mask, circle_mask)
                    white_pixels = cv2.countNonZero(circle_region)
                    total_pixels = cv2.countNonZero(circle_mask)
                    pixel_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
                    
                    # 如果圆内的颜色一致性足够高
                    if pixel_ratio > 0.3:  # 可以调整这个阈值
                        # 画圆
                        cv2.circle(result, center, radius, (0, 255, 0), 2)
                        # 画圆心
                        cv2.circle(result, center, 2, (0, 0, 255), -1)
                        
                        # 显示信息
                        info_text = f"Pos:({center[0]},{center[1]}) R:{radius}"
                        cv2.putText(result, info_text, 
                                  (center[0]-10, center[1]-radius-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # 显示详细信息
                        detail_text = f"Ratio:{pixel_ratio:.2f}"
                        cv2.putText(result, detail_text,
                                  (center[0]-10, center[1]+radius+20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow('Result', result)
            
        except Exception as e:
            print(f"错误：{e}")
            continue
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_pingpong_ball() 