import torch
from IPython.display import Image, clear_output
import os
import yaml
from glob import glob
import subprocess
import platform

def setup_training():
    # 克隆YOLOv5仓库
    if not os.path.exists('yolov5'):
        subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5'])
        # 安装依赖时指定 torch 版本，确保 MPS 支持
        subprocess.run(['pip', 'install', 'torch', 'torchvision', 'torchaudio'])
        subprocess.run(['pip', 'install', '-r', 'yolov5/requirements.txt'])
    
    # 创建数据配置文件
    data_yaml = """
    train: ../dataset/images/train
    val: ../dataset/images/val
    
    nc: 1  # 类别数量
    names: ['pingpong']  # 类别名称
    """
    
    with open('dataset.yaml', 'w') as f:
        f.write(data_yaml)
    
    # 创建模型配置文件（基于yolov5s.yaml修改）
    with open('yolov5/models/yolov5s.yaml', 'r') as f:
        model_yaml = yaml.safe_load(f)
    
    model_yaml['nc'] = 1  # 修改类别数量
    
    with open('custom_model.yaml', 'w') as f:
        yaml.dump(model_yaml, f)

def check_mps_support():
    """检查 MPS 支持状态"""
    if not platform.system() == 'Darwin':
        return False, "非 Mac 系统"
    
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            return False, "PyTorch 未启用 MPS 支持"
        return False, "系统不支持 MPS"
    
    # 验证 MPS 是否真正可用
    try:
        _ = torch.zeros(1).to(torch.device("mps"))
        return True, "MPS 可用且正常工作"
    except Exception as e:
        return False, f"MPS 初始化失败: {str(e)}"

def train_model():
    # 检测系统和设备
    mps_available, mps_status = check_mps_support()
    print(f"MPS 状态: {mps_status}")
    
    # 设置环境变量
    if platform.system() == 'Darwin':
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        # 强制使用 MPS
        os.environ['PYTORCH_DEFAULT_DEVICE'] = 'mps'
    
    # 选择合适的设备
    if mps_available:
        device = 'mps'
        print("✓ 使用 Apple Silicon GPU (MPS) 加速训练")
        # 验证 MPS 是否真正在使用
        test_tensor = torch.zeros(1)
        test_tensor = test_tensor.to('mps')
        print(f"✓ 测试张量设备: {test_tensor.device}")
    elif torch.cuda.is_available():
        device = '0'
        print("✓ 使用 NVIDIA GPU 训练")
    else:
        device = 'cpu'
        print("! 使用 CPU 训练 (性能较低)")
    
    # 根据设备调整批次大小
    if device == 'mps':
        batch_size = 32  # M1/M2 GPU 可以处理更大的批次
    elif device == 'cpu':
        batch_size = 8   # CPU 使用较小批次
    else:
        batch_size = 16  # NVIDIA GPU
    
    print(f"✓ 使用批次大小: {batch_size}")
    
    # 创建超参数文件
    hyp = {
        # 优化器参数
        'lr0': 0.01,              # 初始学习率
        'lrf': 0.01,              # 最终学习率 = lr0 * lrf
        'momentum': 0.937,        # SGD 动量/Adam beta1
        'weight_decay': 0.0005,   # 权重衰减
        'warmup_epochs': 3.0,     # 预热轮数
        'warmup_momentum': 0.8,   # 预热动量
        'warmup_bias_lr': 0.1,    # 预热偏置学习率
        'nbs': 64,                # 标称批次大小
        
        # 损失函数参数
        'box': 0.05,              # 框损失增益
        'cls': 0.5,               # cls损失增益
        'cls_pw': 1.0,            # cls BCELoss正权重
        'obj': 1.0,               # obj损失增益
        'obj_pw': 1.0,            # obj BCELoss正权重
        'iou_t': 0.20,            # IoU训练阈值
        'anchor_t': 4.0,          # 锚点损失阈值
        'fl_gamma': 0.0,          # 焦点损失gamma
        
        # 数据增强参数
        'hsv_h': 0.015,           # 图像HSV-Hue增强
        'hsv_s': 0.7,             # 图像HSV-Saturation增强
        'hsv_v': 0.4,             # 图像HSV-Value增强
        'degrees': 0.0,           # 图像旋转 (+/- deg)
        'translate': 0.1,         # 图像平移 (+/- fraction)
        'scale': 0.5,             # 图像缩放 (+/- gain)
        'shear': 0.0,             # 图像剪切 (+/- deg)
        'perspective': 0.0,       # 图像透视 (+/- fraction)
        'flipud': 0.0,            # 图像上下翻转概率
        'fliplr': 0.5,            # 图像左右翻转概率
        'mosaic': 1.0,            # 图像马赛克概率
        'mixup': 0.0,             # 图像混合概率
        'copy_paste': 0.0,        # 段落复制粘贴概率
        
        # 其他参数
        'giou': 0.05,             # GIoU 损失增益
        'cls_pw': 1.0,            # cls BCELoss 正权重
        'obj_pw': 1.0,            # obj BCELoss 正权重
        'anchor_t': 4.0,          # 锚点-标签匹配阈值
    }
    
    # 保存超参数
    with open('hyp.yaml', 'w') as f:
        yaml.dump(hyp, f)
    
    # 修改 YOLOv5 源码以支持 MPS
    yolo_utils_path = 'yolov5/utils/torch_utils.py'
    if os.path.exists(yolo_utils_path):
        with open(yolo_utils_path, 'r') as f:
            content = f.read()
        
        # 添加 MPS 设备支持
        if 'mps' not in content:
            content = content.replace(
                "device = 'cuda' if torch.cuda.is_available() else 'cpu'",
                "device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'"
            )
            
            with open(yolo_utils_path, 'w') as f:
                f.write(content)
            print("✓ 已修改 YOLOv5 源码以支持 MPS")
    
    # 开始训练
    train_cmd = [
        'python', 'yolov5/train.py',
        '--img', '416',          # 降低图像大小以减少内存使用
        '--batch-size', str(batch_size),
        '--epochs', '300',       # 训练轮数
        '--data', 'dataset.yaml',
        '--cfg', 'custom_model.yaml',
        '--weights', 'yolov5s.pt',
        '--name', 'pingpong_model',
        '--hyp', 'hyp.yaml',     # 使用自定义超参数
        '--patience', '50',      # 早停参数
        '--cache', 'disk',       # 使用磁盘缓存而不是内存缓存
        '--optimizer', 'Adam',   # 使用Adam优化器
        '--label-smoothing', '0.1',
        '--device', device,      # 使用检测到的设备
        '--multi-scale',         # 多尺度训练
        '--exist-ok',           # 允许覆盖已有结果
        '--workers', '4',        # 减少工作线程数以降低内存使用
        '--image-weights',       # 使用图像权重
        '--rect',               # 使用矩形训练
        '--cos-lr',             # 使用余弦学习率调度
        '--sync-bn',            # 使用同步批归一化
        '--noval',              # 禁用验证以节省内存
        '--noautoanchor',       # 禁用自动锚点以节省内存
    ]
    
    # 修改 YOLOv5 源码以优化 MPS 内存使用
    yolo_utils_path = 'yolov5/utils/torch_utils.py'
    if os.path.exists(yolo_utils_path):
        with open(yolo_utils_path, 'r') as f:
            content = f.read()
        
        # 添加内存优化
        if 'mps' in content and 'empty_cache' not in content:
            content = content.replace(
                'def select_device(device=\'\', batch_size=None, newline=True):',
                '''def select_device(device=\'\', batch_size=None, newline=True):
    import gc
    gc.collect()
    if device == \'mps\':
        torch.mps.empty_cache()'''
            )
            
            with open(yolo_utils_path, 'w') as f:
                f.write(content)
            print("✓ 已修改 YOLOv5 源码以优化内存使用")
    
    print("\n开始训练...")
    print(f"✓ 使用设备: {device}")
    print(f"✓ 批次大小: {batch_size}")
    print(f"✓ 内存优化已启用")
    
    # 清理内存
    if device == 'mps':
        import gc
        gc.collect()
        torch.mps.empty_cache()
    
    # 设置环境变量
    env = os.environ.copy()
    if device == 'mps':
        env['PYTORCH_DEFAULT_DEVICE'] = 'mps'
        env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        # 调整 MPS 内存设置
        env['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.5'  # 降低内存使用上限
        env['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.3'   # 设置内存释放阈值
        
        # 降低批次大小和图像尺寸
        batch_size = 8  # 进一步降低批次大小
        
        # 设置保守的内存池大小
        env['PYTORCH_MPS_MEMORY_POOL_SIZE'] = '2048'  # 2GB 内存池大小
        
        print("✓ MPS 内存优化设置:")
        print(f"  - 高水位比例: 50%")
        print(f"  - 低水位比例: 30%")
        print(f"  - 内存池大小: 2GB")
        print(f"  - 批次大小: {batch_size}")
    
    # 修改训练命令
    train_cmd = [
        'python', 'yolov5/train.py',
        '--img', '384',          # 进一步降低图像大小
        '--batch-size', str(batch_size),
        '--epochs', '300',       # 训练轮数
        '--data', 'dataset.yaml',
        '--cfg', 'custom_model.yaml',
        '--weights', 'yolov5s.pt',
        '--name', 'pingpong_model',
        '--hyp', 'hyp.yaml',     # 使用自定义超参数
        '--patience', '50',      # 早停参数
        '--cache', 'disk',       # 使用磁盘缓存
        '--optimizer', 'Adam',   # 使用Adam优化器
        '--label-smoothing', '0.1',
        '--device', device,
        '--multi-scale',
        '--exist-ok',
        '--workers', '2',        # 进一步减少工作线程
        '--image-weights',
        '--rect',
        '--cos-lr',
        '--sync-bn',
        '--noval',              # 禁用验证
        '--noautoanchor',       # 禁用自动锚点
        '--nosave',             # 不保存中间结果
    ]
    
    # 在训练前清理内存
    if device == 'mps':
        import gc
        gc.collect()
        torch.mps.empty_cache()
        
        # 等待一会儿让系统释放内存
        print("等待系统释放内存...")
        import time
        time.sleep(5)
    
    print("\n开始训练...")
    print(f"✓ 使用设备: {device}")
    print(f"✓ 批次大小: {batch_size}")
    print(f"✓ 图像尺寸: 384")
    
    # 使用更新后的环境变量运行命令
    subprocess.run(train_cmd, env=env)

if __name__ == "__main__":
    print("系统检查:")
    print(f"✓ 操作系统: {platform.system()}")
    print(f"✓ Python版本: {platform.python_version()}")
    print(f"✓ PyTorch版本: {torch.__version__}")
    
    # 检查 MPS 支持
    mps_available, mps_status = check_mps_support()
    if mps_available:
        print("✓ MPS 加速可用")
    else:
        print(f"! {mps_status}")
    
    # 确保安装了正确的 PyTorch 版本
    if platform.system() == 'Darwin' and not mps_available:
        print("\n要启用 MPS 加速，请运行:")
        print("pip install --upgrade torch torchvision torchaudio")
    
    setup_training()
    train_model()