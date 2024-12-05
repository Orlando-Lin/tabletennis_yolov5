import os
import cv2
import numpy as np
from glob import glob
import requests
import shutil
from tqdm import tqdm
import albumentations as A
from PIL import Image
import xml.etree.ElementTree as ET

def create_dataset_structure():
    """创建数据集目录结构"""
    directories = [
        'dataset/images/train',
        'dataset/images/val',
        'dataset/labels/train',
        'dataset/labels/val',
        'dataset/augmented/images',
        'dataset/augmented/labels'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")

def capture_from_camera():
    """从摄像头捕获图像"""
    cap = cv2.VideoCapture(0)
    
    # 获取已有图片的最大编号
    existing_images = glob('dataset/images/train/camera_*.jpg')
    if existing_images:
        max_num = max([int(img.split('_')[-1].split('.')[0]) for img in existing_images])
        image_count = max_num + 1
    else:
        image_count = 0
    
    print("按's'保存图像，按'q'退出")
    print(f"当前图片编号从 {image_count:04d} 开始")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 显示当前编号
        info_text = f"Next image will be: camera_{image_count:04d}.jpg"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Capture', frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            # 保存图像
            image_path = f'dataset/images/train/camera_{image_count:04d}.jpg'
            cv2.imwrite(image_path, frame)
            print(f"保存图像: {image_path}")
            image_count += 1
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n共捕获 {image_count} 张新图片")

def auto_label_images():
    """使用OpenCV自动标注乒乓球"""
    images = glob('dataset/images/train/*.jpg')
    
    for image_path in tqdm(images, desc="自动标注"):
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            continue
            
        height, width = img.shape[:2]
        
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 定义乒乓球的颜色范围（白色和橙色）
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        lower_orange = np.array([5, 100, 100])
        upper_orange = np.array([25, 255, 255])
        
        # 创建掩码
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        mask = cv2.bitwise_or(mask_white, mask_orange)
        
        # 进行形态学操作
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建标注文件
        label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt')
        
        # 创建可视化图像
        vis_img = img.copy()
        found_balls = False
        
        with open(label_path, 'w') as f:
            for contour in contours:
                # 计算圆形度
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                if area > 100 and circularity > 0.7:
                    # 获取边界框
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 转换为YOLO格式（归一化坐标）
                    x_center = (x + w/2) / width
                    y_center = (y + h/2) / height
                    w_norm = w / width
                    h_norm = h / height
                    
                    # 写入标注文件
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                    
                    # 在可视化图像上绘制边界框和信息
                    cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.circle(vis_img, (x+w//2, y+h//2), 2, (0, 0, 255), -1)
                    cv2.putText(vis_img, f"Ball: {circularity:.2f}", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    found_balls = True
        
        # 显示标注结果
        cv2.imshow('Original', img)
        cv2.imshow('Mask', mask)
        cv2.imshow('Detection', vis_img)
        
        # 等待按键
        key = cv2.waitKey(1)
        if key == ord('q'):  # 按q退出
            break
        elif key == ord('s'):  # 按s跳过当前图像
            continue
    
    cv2.destroyAllWindows()
    print("\n标注完成！")
    print("标注文件格式说明：")
    print("每行格式: <class_id> <x_center> <y_center> <width> <height>")
    print("- class_id: 0 (乒乓球)")
    print("- 所有坐标都已归一化到0-1范围")
    print("- 每个.txt文件对应一张图像")

def manual_labeling():
    """手动标注乒乓球"""
    # 获取所有图片
    train_images = glob('dataset/images/train/*.jpg')
    val_images = glob('dataset/images/val/*.jpg')
    
    # 如果验证集为空，从训练集中分配
    if not val_images and len(train_images) >= 5:  # 确保有足够的图片来分割
        print("\n创建验证集...")
        total_images = len(train_images)
        val_size = int(total_images * 0.2)  # 20%作为验证集
        
        # 随机选择图片
        np.random.shuffle(train_images)
        val_images = train_images[:val_size]
        train_images = train_images[val_size:]
        
        # 移动文件到验证集目录
        for img_path in val_images:
            new_img_path = img_path.replace('/train/', '/val/')
            os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
            shutil.move(img_path, new_img_path)
            
            # 如果已有标注文件，也移动它
            label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
            if os.path.exists(label_path):
                new_label_path = label_path.replace('/train/', '/val/')
                os.makedirs(os.path.dirname(new_label_path), exist_ok=True)
                shutil.move(label_path, new_label_path)
        
        print(f"移动了 {len(val_images)} 张图片到验证集")
    
    # 重新获取所有图片（包括验证集中的图片）
    train_images = glob('dataset/images/train/*.jpg')
    val_images = glob('dataset/images/val/*.jpg')
    all_images = train_images + val_images
    
    # 获取未标注的图片
    def get_unlabeled_images(image_list):
        unlabeled = []
        for img_path in image_list:
            label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
            if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:  # 检查文件是否为空
                unlabeled.append(img_path)
        return unlabeled
    
    # 获取训练集和验证集中未标注的图片
    unlabeled_train = get_unlabeled_images(train_images)
    unlabeled_val = get_unlabeled_images(val_images)
    
    # 合并所有未标注的图片
    images = unlabeled_train + unlabeled_val
    
    # 显示数据集统计信息
    total_images = len(all_images)
    total_unlabeled = len(images)
    val_ratio = len(val_images) / total_images if total_images > 0 else 0
    
    print("\n数据集统计：")
    print(f"训练集总数：{len(train_images)} 张")
    print(f"验证集总数：{len(val_images)} 张")
    print(f"验证集比例：{val_ratio:.1%}")
    print(f"\n待标注统计：")
    print(f"训练集未标注：{len(unlabeled_train)}/{len(train_images)}")
    print(f"验证集未标注：{len(unlabeled_val)}/{len(val_images)}")
    print(f"总计待标注：{total_unlabeled}/{total_images}")
    
    if not images:
        print("\n所有图片都已标注完成！")
        return
    
    # 对未标注的图片进行排序，确保训练集和验证集的图片都能被标注
    images.sort(key=lambda x: ('val' in x, x))  # 先标注训练集，再标注验证集
    
    # 初始化变量
    current_box = []
    drawing = False
    current_image = None
    original_image = None
    image_index = 0
    
    def create_control_panel():
        """创建控制面板"""
        def get_button_positions(panel_width):
            button_width = 140
            spacing = (panel_width - button_width * 5) // 6
            positions = []
            x = spacing
            for i in range(5):
                positions.append((x, 30))
                x += button_width + spacing
            return positions

        if current_image is not None:
            panel_width = current_image.shape[1]
        else:
            panel_width = 800

        panel = np.zeros((100, panel_width, 3), dtype=np.uint8)
        button_positions = get_button_positions(panel_width)
        
        # 使用英文按钮文字，避免中文乱码
        buttons = [
            (b"Save", button_positions[0]),
            (b"Clear", button_positions[1]),
            (b"Prev", button_positions[2]),
            (b"Next", button_positions[3]),
            (b"Delete", button_positions[4])
        ]

        # 绘制按钮
        for text, pos in buttons:
            # 绘制按背景
            cv2.rectangle(panel, (pos[0], pos[1]-25), (pos[0]+140, pos[1]+25), 
                        (0, 255, 0), 2)
            
            # 计算文本位置使其居中
            text_size = cv2.getTextSize(text.decode(), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = pos[0] + (140 - text_size[0]) // 2
            text_y = pos[1] + text_size[1] // 2
            cv2.putText(panel, text.decode(), (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return panel

    def check_button_click(x, y, panel_height):
        """检查是���点击了按钮"""
        if y >= panel_height:  # 点击在控制面板区域
            y = y - panel_height  # 调整y坐标到面板坐标系
            panel_width = current_image.shape[1]
            button_width = 140
            spacing = (panel_width - button_width * 5) // 6
            
            # 计算每个按钮的位置
            buttons = {}
            x_pos = spacing
            for i, action in enumerate(['save', 'clear', 'prev', 'next', 'delete']):
                # 定义按钮的点击区域（上角和右下角的坐标）
                buttons[action] = {
                    'x1': x_pos,
                    'x2': x_pos + button_width,
                    'y1': 5,
                    'y2': 55
                }
                x_pos += button_width + spacing
            
            # 检查点击位置
            for action, coords in buttons.items():
                if (coords['x1'] <= x <= coords['x2'] and 
                    coords['y1'] <= y <= coords['y2']):
                    print(f"Button clicked: {action}")  # 调试信息
                    return action
        return None

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, current_box, current_image, original_image
        
        try:
            # 确保图像已加载
            if current_image is None:
                return
            
            panel_height = current_image.shape[0]
            temp_image = current_image.copy()
            
            # 在图像区域内显示十字辅助线
            if y < panel_height:
                # 绘制贯穿整个图像的十字线
                cursor_color = (0, 255, 0)  # 绿色
                cursor_thickness = 1
                
                # 水平线和垂直线（贯穿整个图像）
                cv2.line(temp_image, (0, y), (temp_image.shape[1], y), 
                        cursor_color, cursor_thickness)
                cv2.line(temp_image, (x, 0), (x, temp_image.shape[0]), 
                        cursor_color, cursor_thickness)
                
                # 显示坐标信息
                coord_text = f"({x}, {y})"
                cv2.putText(temp_image, coord_text, (x + 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, cursor_color, 1)
                
                # 如果正在画框，显示实时的矩形
                if drawing and len(current_box) == 1:
                    start_x, start_y = current_box[0]
                    rect_x1 = min(start_x, x)
                    rect_y1 = min(start_y, y)
                    rect_x2 = max(start_x, x)
                    rect_y2 = max(start_y, y)
                    
                    # 绘制矩形框
                    cv2.rectangle(temp_image, (rect_x1, rect_y1), (rect_x2, rect_y2), 
                                (0, 255, 0), 2)
                    
                    # 显示尺寸信息
                    w = abs(x - start_x)
                    h = abs(y - start_y)
                    size_text = f"{w}x{h}"
                    cv2.putText(temp_image, size_text, (rect_x1, rect_y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 更新显示
            panel = create_control_panel()
            combined_image = np.vstack([temp_image, panel])
            cv2.imshow('Manual Labeling', combined_image)
            cv2.waitKey(1)  # 重要：确保窗口更新
            
            # 处理鼠标事件
            if event == cv2.EVENT_LBUTTONDOWN:
                if y < panel_height:  # 在图像区域内
                    # 检查是否已经有标注
                    label_path = images[image_index].replace('images', 'labels').replace('.jpg', '.txt')
                    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
                        drawing = True
                        current_box = [(x, y)]
                else:  # 在控制面板区域
                    button = check_button_click(x, y, panel_height)
                    if button:
                        handle_button_action(button)
            
            elif event == cv2.EVENT_LBUTTONUP:
                if drawing and y < panel_height:
                    drawing = False
                    if len(current_box) == 1:
                        if abs(x - current_box[0][0]) > 5 and abs(y - current_box[0][1]) > 5:
                            current_box.append((x, y))
                            rect_x1 = min(current_box[0][0], x)
                            rect_y1 = min(current_box[0][1], y)
                            rect_x2 = max(current_box[0][0], x)
                            rect_y2 = max(current_box[0][1], y)
                            cv2.rectangle(current_image, (rect_x1, rect_y1), 
                                        (rect_x2, rect_y2), (0, 255, 0), 2)
                        else:
                            current_box = []
        
        except Exception as e:
            print(f"Mouse callback error: {e}")
            print(f"Event: {event}, x: {x}, y: {y}")
    
    def handle_button_action(action):
        """处理按钮点击事件"""
        nonlocal image_index, current_box, current_image, original_image
        
        button_actions = {
            "save": lambda: save_annotation() if len(current_box) == 2 else None,
            "clear": lambda: clear_annotation(),
            "prev": lambda: navigate_image(-1),
            "next": lambda: navigate_image(1),
            "delete": lambda: delete_annotation()
        }
        
        if action in button_actions:
            button_actions[action]()
    
    def save_annotation():
        """保存标注"""
        if len(current_box) != 2:
            # 如果没有完整的框，显示提示
            temp_image = current_image.copy()
            cv2.putText(temp_image, "No box to save!", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            panel = create_control_panel()
            combined_image = np.vstack([temp_image, panel])
            cv2.imshow('Manual Labeling', combined_image)
            return

        height, width = original_image.shape[:2]
        x1, y1 = current_box[0]
        x2, y2 = current_box[1]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        x_center = (x1 + x2) / 2 / width
        y_center = (y1 + y2) / 2 / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        
        label_path = images[image_index].replace('images', 'labels').replace('.jpg', '.txt')
        with open(label_path, 'a') as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
        
        # 显���保存成功的提示
        temp_image = current_image.copy()
        # 在图像中央显示绿色的成功提示
        success_text = "Annotation Saved!"
        text_size = cv2.getTextSize(success_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height // 2
        
        # 绘制半透明背景
        overlay = temp_image.copy()
        cv2.rectangle(overlay, 
                     (text_x - 10, text_y - 30),
                     (text_x + text_size[0] + 10, text_y + 10),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, temp_image, 0.5, 0, temp_image)
        
        # 绘制文本
        cv2.putText(temp_image, success_text,
                    (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)
        
        # 显示更新后的图像
        panel = create_control_panel()
        combined_image = np.vstack([temp_image, panel])
        cv2.imshow('Manual Labeling', combined_image)
        
        # 等待一小段时间后刷新图像
        cv2.waitKey(500)  # 显示500毫秒
        
        print(f"Saved annotation to: {label_path}")
        current_box.clear()
        load_image()
        
        # 保存后检查是否全部完成
        if check_completion():
            return True  # 表示已完成所有标注
        return False
    
    def clear_annotation():
        """清除当前标注"""
        current_box.clear()
        current_image = original_image.copy()
        load_image()
    
    def navigate_image(direction):
        """导航到上一张/下一张图片"""
        nonlocal image_index
        new_index = image_index + direction
        if 0 <= new_index < len(images):
            image_index = new_index
            load_image()
            # 检查是否是最后一张图片
            if new_index == len(images) - 1:
                check_completion()

    def check_completion():
        """检查是否所有图片都已标注"""
        all_labeled = True
        unlabeled_count = 0
        
        for img_path in images:
            label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
            if not os.path.exists(label_path):
                all_labeled = False
                unlabeled_count += 1
        
        if all_labeled:
            # 显示完成提示
            temp_image = current_image.copy()
            complete_text = "All images labeled! Saving and exiting..."
            text_size = cv2.getTextSize(complete_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = (temp_image.shape[1] - text_size[0]) // 2
            text_y = temp_image.shape[0] // 2
            
            # 绘制半透明背景
            overlay = temp_image.copy()
            cv2.rectangle(overlay, 
                         (text_x - 10, text_y - 30),
                         (text_x + text_size[0] + 10, text_y + 10),
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, temp_image, 0.5, 0, temp_image)
            
            # 绘制文本
            cv2.putText(temp_image, complete_text,
                       (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, (0, 255, 0), 2)
            
            panel = create_control_panel()
            combined_image = np.vstack([temp_image, panel])
            cv2.imshow('Manual Labeling', combined_image)
            cv2.waitKey(1500)  # 显示1.5秒
            
            print("\n标注完成统计：")
            print(f"总图片数量: {len(images)}")
            print(f"已标注数量: {len(images)}")
            print("所有图片已完成标注")
            
            cv2.destroyAllWindows()
            return True
        else:
            # 显示未完成提示
            remaining_text = f"Remaining: {unlabeled_count} images"
            cv2.putText(current_image, remaining_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return False
    
    def delete_annotation():
        """删除当前图片的标注"""
        label_path = images[image_index].replace('images', 'labels').replace('.jpg', '.txt')
        if os.path.exists(label_path):
            os.remove(label_path)
            print(f"Deleted annotation file: {label_path}")
        load_image()
    
    def load_image():
        """加载并显示当前图像"""
        nonlocal current_image, original_image, current_box, image_index
        
        try:
            if image_index >= len(images):
                print("已到达最后一张图片")
                return
            
            current_box = []
            original_image = cv2.imread(images[image_index])
            
            if original_image is None:
                print(f"无法读取图像: {images[image_index]}")
                image_index += 1
                if image_index < len(images):
                    load_image()
                return
            
            current_image = original_image.copy()
            height, width = current_image.shape[:2]
            
            # 显示已有的标注
            label_path = images[image_index].replace('images', 'labels').replace('.jpg', '.txt')
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        class_id, x_center, y_center, w, h = map(float, line.strip().split())
                        x = int((x_center - w/2) * width)
                        y = int((y_center - h/2) * height)
                        w = int(w * width)
                        h = int(h * height)
                        cv2.rectangle(current_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 显示图像信息和进度（调整位置避免重叠）
            info_text = f"Image: {os.path.basename(images[image_index])}"
            count_text = f"({image_index + 1}/{len(images)})"
            progress_text = f"Progress: {((image_index + 1)/len(images))*100:.1f}%"
            
            # 计算文本大小
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            # 添加半透明背景
            overlay = current_image.copy()
            bg_padding = 5
            
            # 第一行：图像名称
            (text_width, text_height), baseline = cv2.getTextSize(info_text, font, font_scale, thickness)
            cv2.rectangle(overlay, (10-bg_padding, 30-text_height-bg_padding), 
                         (10+text_width+bg_padding, 30+bg_padding), 
                         (0, 0, 0), -1)
            cv2.putText(current_image, info_text, (10, 30), font, font_scale, (0, 255, 0), thickness)
            
            # 第二行：计数信息
            y_offset = 60
            (text_width, text_height), baseline = cv2.getTextSize(count_text, font, font_scale, thickness)
            cv2.rectangle(overlay, (10-bg_padding, y_offset-text_height-bg_padding), 
                         (10+text_width+bg_padding, y_offset+bg_padding), 
                         (0, 0, 0), -1)
            cv2.putText(current_image, count_text, (10, y_offset), font, font_scale, (0, 255, 0), thickness)
            
            # 第三行：进度信息
            y_offset = 90
            (text_width, text_height), baseline = cv2.getTextSize(progress_text, font, font_scale, thickness)
            cv2.rectangle(overlay, (10-bg_padding, y_offset-text_height-bg_padding), 
                         (10+text_width+bg_padding, y_offset+bg_padding), 
                         (0, 0, 0), -1)
            cv2.putText(current_image, progress_text, (10, y_offset), font, font_scale, (0, 255, 0), thickness)
            
            # 添加半透明效果
            cv2.addWeighted(overlay, 0.3, current_image, 0.7, 0, current_image)
            
            # 添加控制面板
            panel = create_control_panel()
            combined_image = np.vstack([current_image, panel])
            
            # 显示图像
            cv2.imshow('Manual Labeling', combined_image)
            cv2.waitKey(1)
            
        except Exception as e:
            print(f"加载图像时出错: {e}")
            print(f"当前图像索引: {image_index}")
            print(f"总图像数量: {len(images)}")
            print(f"当前图像路径: {images[image_index] if image_index < len(images) else 'N/A'}")
    
    # 在开始标注前检查图片列表
    if not images:
        print("\n没有找到需要标注的图片！")
        print("请确保 dataset/images/train 或 dataset/images/val 目录中有图片文件。")
        return
    
    print(f"\n开始标注，共 {len(images)} 张图片待标注")
    print("操作说明：")
    print("1. 用鼠标拖动画框标注乒乓球")
    print("2. 点击 Save 保存当前标注")
    print("3. 点击 Next/Prev 切换图片")
    print("4. 点击 Clear 清除当前标注")
    print("5. 点击 Delete 删除已有标注")
    print("6. 按 'q' 键退出程序")

    # 初始化变量
    current_box = []
    drawing = False
    current_image = None
    original_image = None
    image_index = 0  # 确保在调用 load_image 前初始化
    
    # 创建窗口和设置回调
    cv2.namedWindow('Manual Labeling', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Manual Labeling', 1280, 1060)
    cv2.setMouseCallback('Manual Labeling', mouse_callback)
    
    # 加载第一张图片
    if len(images) > 0:
        load_image()
        check_completion()  # 初始检查
    
    # 创建窗口
    cv2.namedWindow('Manual Labeling', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Manual Labeling', 1280, 1060)
    cv2.setMouseCallback('Manual Labeling', mouse_callback)
    
    # 添加窗口状态标志
    window_closed = False
    
    while not window_closed:
        try:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # 确保窗口保持打开状态
            if cv2.getWindowProperty('Manual Labeling', cv2.WND_PROP_VISIBLE) < 1:
                break
                
        except Exception as e:
            print(f"Error in main loop: {e}")
            break

    cv2.destroyAllWindows()
    
    # 如果是正常退出，显示最终统计
    if not window_closed:
        labeled_count = sum(1 for img in images if os.path.exists(
            img.replace('images', 'labels').replace('.jpg', '.txt')))
        print("\n最终标注统计：")
        print(f"总图片数量: {len(images)}")
        print(f"已标注数量: {labeled_count}")
        print(f"未标注数量: {len(images) - labeled_count}")

def augment_dataset():
    """数据增强"""
    transform = A.Compose([
        # 基础增强
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.4),
        
        # 几何变换
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=45,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.7
        ),
        
        # 色彩增强
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.5
        ),
        A.RandomBrightness(limit=0.2, p=0.5),
        A.RandomContrast(limit=0.2, p=0.5),
        
        # 添加噪声和模糊
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=7, p=1.0),
            A.GaussianBlur(blur_limit=7, p=1.0),
        ], p=0.3),
        
        # 添加天气效果
        A.OneOf([
            A.RandomShadow(p=1.0),
            A.RandomFog(p=1.0),
            A.RandomBrightness(p=1.0),
        ], p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    images = glob('dataset/images/train/*.jpg')
    
    # 对每张图片进行多次增强
    for idx, image_path in enumerate(tqdm(images, desc="数据增强")):
        image = cv2.imread(image_path)
        label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt')
        
        if not os.path.exists(label_path):
            continue
            
        # 读取标注
        bboxes = []
        class_labels = []
        with open(label_path, 'r') as f:
            for line in f:
                class_id, x_center, y_center, w, h = map(float, line.strip().split())
                bboxes.append([x_center, y_center, w, h])
                class_labels.append(class_id)
        
        # 每张图片生成5个增强版本
        for aug_idx in range(5):
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            
            # 保存增强后的图像
            aug_image_path = f'dataset/augmented/images/aug_{idx:04d}_{aug_idx}.jpg'
            cv2.imwrite(aug_image_path, augmented['image'])
            
            # 保存标注
            aug_label_path = f'dataset/augmented/labels/aug_{idx:04d}_{aug_idx}.txt'
            with open(aug_label_path, 'w') as f:
                for bbox, class_id in zip(augmented['bboxes'], augmented['class_labels']):
                    f.write(f"{int(class_id)} {' '.join(map(str, bbox))}\n")

def convert_voc_to_yolo():
    """将VOC格式的数据集转换为YOLO格式"""
    def convert_box(size, box):
        dw = 1.0/size[0]
        dh = 1.0/size[1]
        x = (box[0] + box[2])/2.0
        y = (box[1] + box[3])/2.0
        w = box[2] - box[0]
        h = box[3] - box[1]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return (x,y,w,h)

    def convert_voc_annotation(xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        boxes = []
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls != 'pingpong':
                continue
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
            bb = convert_box((w,h), b)
            boxes.append(bb)
        return boxes

    # 转换VOC格式的数据
    voc_annotations = glob('dataset/VOC/Annotations/*.xml')
    for xml_path in tqdm(voc_annotations, desc="转换VOC数据"):
        # 获取对应的图像
        img_path = xml_path.replace('Annotations', 'JPEGImages').replace('.xml', '.jpg')
        if not os.path.exists(img_path):
            continue
            
        # 转换标注
        boxes = convert_voc_annotation(xml_path)
        
        # 保存YOLO格式的标注
        yolo_label_path = os.path.basename(xml_path).replace('.xml', '.txt')
        yolo_label_path = os.path.join('dataset/labels/train', yolo_label_path)
        
        with open(yolo_label_path, 'w') as f:
            for box in boxes:
                f.write(f"0 {' '.join(map(str, box))}\n")
        
        # 复制图像
        shutil.copy(img_path, os.path.join('dataset/images/train', os.path.basename(img_path)))

def split_dataset():
    """分割数据集为训练集和验证集"""
    # 获取所有图片
    train_images = glob('dataset/images/train/*.jpg')
    val_images = glob('dataset/images/val/*.jpg')
    
    # 计算当前的比例
    total_images = len(train_images) + len(val_images)
    current_val_ratio = len(val_images) / total_images if total_images > 0 else 0
    target_val_ratio = 0.2  # 目标验证集比例
    
    print("\n调整验证集比例...")
    print(f"当前验证集比例: {current_val_ratio:.1%}")
    print(f"目标验证集比例: {target_val_ratio:.1%}")
    
    if current_val_ratio < target_val_ratio:
        # 需要移动更多图片到验证集
        needed_val_count = int(total_images * target_val_ratio) - len(val_images)
        if needed_val_count > 0:
            print(f"需要移动 {needed_val_count} 张图片到验证集")
            # 随机选择片
            np.random.shuffle(train_images)
            move_images = train_images[:needed_val_count]
            
            # 移动到验证集
            for img_path in move_images:
                # 移动图像
                new_img_path = img_path.replace('/train/', '/val/')
                os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
                shutil.move(img_path, new_img_path)
                
                # 移动标注文件
                label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
                if os.path.exists(label_path):
                    new_label_path = label_path.replace('/train/', '/val/')
                    os.makedirs(os.path.dirname(new_label_path), exist_ok=True)
                    shutil.move(label_path, new_label_path)
    
    elif current_val_ratio > target_val_ratio:
        # 需要移动一些证集图片回训练集
        excess_val_count = len(val_images) - int(total_images * target_val_ratio)
        if excess_val_count > 0:
            print(f"需要移动 {excess_val_count} 张图片回训练集")
            # 随机选择图片
            np.random.shuffle(val_images)
            move_images = val_images[:excess_val_count]
            
            # 移动回训练集
            for img_path in move_images:
                # 移动图像
                new_img_path = img_path.replace('/val/', '/train/')
                os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
                shutil.move(img_path, new_img_path)
                
                # 移动标注文件
                label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
                if os.path.exists(label_path):
                    new_label_path = label_path.replace('/val/', '/train/')
                    os.makedirs(os.path.dirname(new_label_path), exist_ok=True)
                    shutil.move(label_path, new_label_path)
    
    # 重新计算最终比例
    train_images = glob('dataset/images/train/*.jpg')
    val_images = glob('dataset/images/val/*.jpg')
    total_images = len(train_images) + len(val_images)
    final_val_ratio = len(val_images) / total_images if total_images > 0 else 0
    
    print("\n数据集分割完成：")
    print(f"训练集数量: {len(train_images)}")
    print(f"验证集数量: {len(val_images)}")
    print(f"验证集比例: {final_val_ratio:.1%}")

def main():
    # 创建目录结构
    create_dataset_structure()
    
    print("\n请选择操作：")
    print("1. 从摄像头捕获")
    print("2. 自动标注")
    print("3. 手动标注")
    print("4. 数据增强")
    print("5. 转换VOC数据集")
    print("6. 执行所有操作")
    
    choice = input("请输入选择（1/2/3/4/5/6）: ")
    
    if choice in ['1', '6']:
        capture_from_camera()
    if choice in ['2', '6']:
        auto_label_images()
    if choice in ['3', '6']:
        manual_labeling()
    if choice in ['4', '6']:
        augment_dataset()
    if choice in ['5', '6']:
        convert_voc_to_yolo()
    
    # 分割数据集
    split_dataset()
    
    print("\n数据集准备完成！")
    print("请检查标注结果，必要时使用labelImg手动修正")

if __name__ == "__main__":
    main() 