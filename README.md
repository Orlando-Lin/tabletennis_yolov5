# 乒乓球检测训练系统+数据标注工具(prepare_dataset)

这是一个基于 YOLOv5 的乒乓球检测系统，支持数据采集、标注和模型训练。特别优化了在 Apple Silicon (M1/M2) Mac 上的训练性能。

python的数据标注工具prepare_dataset.py文件。

## 功能特点

1. 数据采集
   - 支持摄像头实时采集
   - 自动编号和保存
   - 支持多种图像格式

2. 数据标注
   - 手动标注工具
   - 自动标注功能
   - 实时预览和编辑
   - 支持标注验证

3. 数据增强
   - 自动数据增强
   - 多种增强方式
   - 自动保存增强结果

4. 模型训练
   - 支持 Apple Silicon GPU 加速
   - 自动内存优化
   - 训练进度可视化
   - 支持断点续训

## 系统要求

- Python 3.8 或更高版本
- PyTorch 2.0 或更高版本
- OpenCV
- 摄像头（用于数据采集）
- 足够的存储空间（建议 10GB 以上）

## 安装步骤

1. 克隆仓库：

```bash
git clone [https://github.com/Orlando-Lin/tabletennis_yolov5.git]
cd [项目目录]
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 安装 PyTorch（Mac M1/M2）：

```bash
pip install --upgrade torch torchvision torchaudio
```

## 使用说明

### 1. 启动程序

```bash
python prepare_dataset.py
```

### 2. 选择操作模式
程序提供以下操作选项：
1. 从摄像头捕获
2. 自动标注
3. 手动标注
4. 数据增强
5. 转换 VOC 数据集
6. 执行所有操作

### 3. 数据采集
- 运行程序选择选项 1
- 按 's' 保存图像
- 按 'q' 退出采集

### 4. 手动标注
- 运行程序选择选项 3
- 使用鼠标拖动绘制标注框
- 使用界面按钮进行操作：
  - Save: 保存当前标注
  - Clear: 清除当前标注
  - Prev/Next: 切换图片
  - Delete: 删除标注

### 5. 训练模型

```bash
python train_model.py
```

训练过程会自动：
- 选择合适的设备（MPS/CPU）
- 优化内存使用
- 显示训练进度
- 保存训练结果

### 6. 查看训练进度

```bash
tensorboard --logdir yolov5/runs/train
```
访问 http://localhost:6006 查看训练曲线

## 文件结构
```
.
├── prepare_dataset.py    # 数据准备主程序
├── train_model.py        # 模型训练程序
├── dataset/             # 数据集目录
│   ├── images/         # 图像文件
│   └── labels/         # 标注文件
├── yolov5/             # YOLOv5 目录
└── requirements.txt    # 依赖列表
```

## 注意事项

1. 数据采集
   - 确保光线充足
   - 保持摄像头稳定
   - 采集不同角度的图像

2. 标注要求
   - 标注框要紧贴乒乓球边缘
   - 确保标注准确性
   - 检查标注质量

3. 训练优化
   - Mac 用户建议使用 MPS 加速
   - 根据内存大小调整批次大小
   - 适当调整图像尺寸

4. 常见问题
   - 如果出现内存不足，尝试减小批次大小
   - 如果训练速度慢，可以减小图像尺寸
   - 如果标注不准确，增加训练轮数

## 更新日志

- v1.0.0
  - 初始版本发布
  - 支持基本功能
  - 添加 Mac M1/M2 支持

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目。

## 许可证

本项目采用 MIT 许可证。

## 联系方式

如有问题，请通过以下方式联系：
- Email: [Cao_mouiller@163.com]
- GitHub: [https://github.com/Orlando-Lin]
- gitee: [https://gitee.com/orlando_lin]