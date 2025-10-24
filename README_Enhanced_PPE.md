# 🚀 Enhanced PPE Detection System

A state-of-the-art Personal Protective Equipment (PPE) detection system using **YOLOv8m** with advanced computer vision techniques for high-accuracy detection of safety equipment in construction and industrial environments.

## 🎯 Features

### 🔧 **Model Upgrades**
- **YOLOv8m Architecture**: Medium-sized model for optimal speed/accuracy balance
- **High Resolution**: 960px image size for better small-object detection
- **Extended Training**: 100 epochs with pretrained weights
- **Optimized Inference**: conf=0.25, iou=0.45 for precise detection

### 🧠 **Enhanced Detection Classes**
- **Helmet** 🪖: Hard hats and safety helmets
- **Vest** 🦺: High-visibility safety vests
- **Gloves** 🧤: Work gloves and safety gloves  
- **Glasses** 👓: Safety glasses and goggles

### 🏋️‍♂️ **Advanced Training**
- **Balanced Dataset**: Construction PPE Detection (v3) from Roboflow
- **Smart Augmentations**: Rotation ±15°, brightness ±25%, horizontal flip 50%, scale 0.5-1.5
- **Optimized Parameters**: AdamW optimizer, learning rate scheduling, weight decay

### ⚡ **FastAPI Backend**
- **High-Performance API**: Async processing with FastAPI
- **Multiple Endpoints**: `/detect`, `/batch-detect`, `/health`, `/model-info`
- **Image Enhancement**: Multiple preprocessing strategies
- **Smart Filtering**: Duplicate removal, IoU-based merging

### 🎨 **Modern Web Interface**
- **Responsive Design**: Mobile-friendly interface
- **Real-time Detection**: Live image processing
- **Visual Feedback**: Annotated results with confidence scores
- **Compliance Status**: Clear YES/NO compliance reporting

## 🚀 Quick Start

### 1. **Setup Environment**
```bash
# Clone or download the project
cd enhanced-ppe-detection

# Run automated setup
python setup_enhanced_ppe.py
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Download Dataset**
```python
# Get your Roboflow API key from: https://app.roboflow.com/settings/api
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("construction-ppe").project("construction-ppe-detection")
dataset = project.version(3).download("yolov8")

# Move to datasets/ppe-balanced/
```

### 4. **Train the Model**
```bash
python quick_train.py
```

### 5. **Start the API**
```bash
python start_api.py
```

### 6. **Access the System**
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📊 Model Performance

### **Training Configuration**
```yaml
Model: YOLOv8m
Image Size: 960px
Epochs: 100
Batch Size: 16
Optimizer: AdamW
Learning Rate: 0.01
Weight Decay: 0.0005
```

### **Inference Parameters**
```yaml
Confidence: 0.25
IoU Threshold: 0.45
Max Detections: 300
Image Size: 960px
```

### **Expected Performance**
- **mAP50**: >0.85
- **mAP50-95**: >0.65
- **Inference Speed**: ~50ms per image (GPU)
- **Accuracy**: >95% for clear images

## 🔧 API Endpoints

### **POST /detect**
Upload an image for PPE detection
```python
import requests

files = {'file': open('image.jpg', 'rb')}
response = requests.post('http://localhost:8000/detect', files=files)
result = response.json()
```

**Response:**
```json
{
  "success": true,
  "detections": [
    {
      "class": "helmet",
      "confidence": 0.95,
      "bbox": [100, 50, 200, 150],
      "strategy": "standard"
    }
  ],
  "ppe_counts": {
    "helmet": 1,
    "vest": 1,
    "gloves": 0,
    "glasses": 1
  },
  "compliance_status": "NON-COMPLIANT",
  "missing_ppe": ["gloves"],
  "annotated_image": "/static/detections/image.jpg"
}
```

### **POST /batch-detect**
Process multiple images (max 10)
```python
files = [
    ('files', open('image1.jpg', 'rb')),
    ('files', open('image2.jpg', 'rb'))
]
response = requests.post('http://localhost:8000/batch-detect', files=files)
```

### **GET /health**
Check system health
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "runs/detect/ppe_detector_v2/weights/best.pt"
}
```

## 🐳 Docker Deployment

### **Build and Run**
```bash
# Build the image
docker build -t enhanced-ppe-detection .

# Run with GPU support
docker run --gpus all -p 8000:8000 enhanced-ppe-detection

# Or use docker-compose
docker-compose up -d
```

### **Docker Compose**
```yaml
version: '3.8'
services:
  ppe-detection:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./static:/app/static
      - ./runs:/app/runs
    environment:
      - CUDA_VISIBLE_DEVICES=0
```

## 📁 Project Structure

```
enhanced-ppe-detection/
├── app.py                          # FastAPI backend
├── train_ppe.py                    # Training script
├── quick_train.py                  # Quick training
├── start_api.py                    # API startup
├── setup_enhanced_ppe.py           # Setup script
├── requirements.txt                # Dependencies
├── data.yaml                       # Dataset config
├── Dockerfile                      # Docker config
├── docker-compose.yml             # Docker compose
├── templates/
│   └── index.html                 # Web interface
├── static/
│   ├── uploads/                   # Upload directory
│   ├── detections/                # Detection results
│   └── results/                   # Processing results
├── datasets/
│   └── ppe-balanced/              # Training dataset
│       ├── train/
│       ├── valid/
│       └── test/
└── runs/
    └── detect/
        └── ppe_detector_v2/       # Training outputs
```

## 🎯 Use Cases

### **Construction Sites**
- Real-time safety compliance monitoring
- Automated safety reporting
- Worker protection verification

### **Industrial Facilities**
- PPE compliance checking
- Safety protocol enforcement
- Risk assessment automation

### **Training & Education**
- Safety training visualization
- Compliance demonstration
- Risk awareness programs

## 🔧 Customization

### **Adding New Classes**
1. Update `data.yaml` with new class names
2. Retrain the model with new dataset
3. Update `CLASS_NAMES` in `app.py`

### **Adjusting Detection Sensitivity**
```python
# In app.py, modify detection parameters
results = model.predict(
    image,
    conf=0.25,      # Lower = more sensitive
    iou=0.45,       # Lower = more detections
    imgsz=960,      # Higher = better accuracy
    max_det=300     # Maximum detections
)
```

### **Custom Augmentations**
```python
# In train_ppe.py, modify augmentation parameters
augmentation_config = {
    'hsv_h': 0.015,     # HSV-Hue
    'hsv_s': 0.7,       # HSV-Saturation  
    'hsv_v': 0.4,       # HSV-Value
    'degrees': 15.0,     # Rotation
    'translate': 0.1,    # Translation
    'scale': 0.5,        # Scale range
    'fliplr': 0.5,       # Horizontal flip
    'mosaic': 1.0,       # Mosaic augmentation
}
```

## 📈 Performance Optimization

### **GPU Acceleration**
```bash
# Install CUDA toolkit
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### **Model Optimization**
```python
# Use TensorRT for inference optimization
model.export(format='engine', device=0)
```

### **Batch Processing**
```python
# Process multiple images efficiently
results = model.predict(images, batch=8)
```

## 🐛 Troubleshooting

### **Common Issues**

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   batch=8  # Instead of 16
   ```

2. **Model Not Loading**
   ```bash
   # Check model path
   ls runs/detect/ppe_detector_v2/weights/
   ```

3. **Dataset Issues**
   ```bash
   # Verify data.yaml structure
   cat data.yaml
   ```

4. **API Connection Issues**
   ```bash
   # Check if API is running
   curl http://localhost:8000/health
   ```

## 📚 Additional Resources

- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Roboflow Dataset**: https://app.roboflow.com/
- **OpenCV Documentation**: https://docs.opencv.org/

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics** for YOLOv8 implementation
- **Roboflow** for Construction PPE Detection dataset
- **FastAPI** for the web framework
- **OpenCV** for computer vision utilities

---

**🚀 Ready to detect PPE with state-of-the-art accuracy!**
