# Table Tennis Ball Detection System

A YOLOv5-based table tennis ball detection system that supports data collection, annotation, and model training. Specially optimized for training on Apple Silicon (M1/M2) Macs.

## Features

1. Data Collection
   - Real-time camera capture
   - Automatic numbering and saving
   - Support for multiple image formats

2. Data Annotation
   - Manual annotation tool
   - Automatic annotation feature
   - Real-time preview and editing
   - Annotation validation support

3. Data Augmentation
   - Automatic data augmentation
   - Multiple augmentation methods
   - Automatic saving of augmented results

4. Model Training
   - Apple Silicon GPU acceleration support
   - Automatic memory optimization
   - Training progress visualization
   - Checkpoint resumption support

## System Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher
- OpenCV
- Camera (for data collection)
- Sufficient storage space (10GB+ recommended)

## Installation

1. Clone the repository:
```bash
git clone [repository URL]
cd [project directory]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install PyTorch (for Mac M1/M2):
```bash
pip install --upgrade torch torchvision torchaudio
```

## Usage Guide

### 1. Launch Program
```bash
python prepare_dataset.py
```

### 2. Select Operation Mode
The program offers the following options:
1. Capture from camera
2. Auto annotation
3. Manual annotation
4. Data augmentation
5. Convert VOC dataset
6. Execute all operations

### 3. Data Collection
- Select option 1 in the program
- Press 's' to save images
- Press 'q' to exit capture mode

### 4. Manual Annotation
- Select option 3 in the program
- Drag mouse to draw annotation boxes
- Use interface buttons:
  - Save: Save current annotation
  - Clear: Clear current annotation
  - Prev/Next: Switch between images
  - Delete: Remove annotation

### 5. Model Training
```bash
python train_model.py
```

The training process automatically:
- Selects appropriate device (MPS/CPU)
- Optimizes memory usage
- Displays training progress
- Saves training results

### 6. Monitor Training Progress
```bash
tensorboard --logdir yolov5/runs/train
```
Visit http://localhost:6006 to view training curves

## Project Structure
```
.
├── prepare_dataset.py    # Data preparation main program
├── train_model.py        # Model training program
├── dataset/             # Dataset directory
│   ├── images/         # Image files
│   └── labels/         # Label files
├── yolov5/             # YOLOv5 directory
└── requirements.txt    # Dependency list
```

## Important Notes

1. Data Collection
   - Ensure adequate lighting
   - Keep camera stable
   - Capture images from different angles

2. Annotation Requirements
   - Bounding boxes should tightly fit the table tennis ball
   - Ensure annotation accuracy
   - Verify annotation quality

3. Training Optimization
   - MPS acceleration recommended for Mac users
   - Adjust batch size based on available memory
   - Optimize image size as needed

4. Troubleshooting
   - Reduce batch size if memory issues occur
   - Decrease image size if training is slow
   - Increase training epochs if accuracy is low

## Memory Optimization for M1/M2 Macs

1. Default Settings:
   - Batch size: 8
   - Image size: 384
   - Memory pool: 2GB
   - Workers: 2

2. Adjustable Parameters:
   - `PYTORCH_MPS_HIGH_WATERMARK_RATIO`: 0.5
   - `PYTORCH_MPS_LOW_WATERMARK_RATIO`: 0.3
   - `PYTORCH_MPS_MEMORY_POOL_SIZE`: 2048

## Training Parameters

1. Basic Settings:
   - Epochs: 300
   - Optimizer: Adam
   - Learning rate: 0.01
   - Label smoothing: 0.1

2. Advanced Options:
   - Multi-scale training
   - Cosine learning rate
   - Synchronized batch normalization

## Changelog

- v1.0.0
  - Initial release
  - Basic functionality
  - Mac M1/M2 support added

## Contributing

Issues and Pull Requests are welcome to improve the project.

## License

This project is licensed under the MIT License.

## Contact

For questions or support:
- Email: [Cao_mouiller@163.com]
- GitHub: [Your GitHub Profile]

## Acknowledgments

- YOLOv5 team for the base detection framework
- PyTorch team for MPS support
- Contributors and testers 