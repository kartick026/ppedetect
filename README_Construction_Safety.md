# 🏗️ Construction Safety PPE Detection System

**Enhanced PPE Detection using Construction Safety Dataset**
- **Dataset**: `kartick025/construction-safety-n0gkb-vd2dp-instant-1`
- **Model**: YOLOv8m with 960px resolution
- **Classes**: Helmet, Vest, Gloves, Glasses

## 🎯 Dataset Information

### **Construction Safety Dataset**
- **Source**: Roboflow - kartick025/construction-safety
- **Version**: 1
- **Format**: YOLOv8
- **Classes**: 4 (helmet, vest, gloves, glasses)
- **Quality**: High-quality construction site images
- **Annotations**: Professional safety equipment labeling

### **Dataset Classes**
1. **Helmet** 🪖: Hard hats and safety helmets
2. **Vest** 🦺: High-visibility safety vests
3. **Gloves** 🧤: Work gloves and safety gloves
4. **Glasses** 👓: Safety glasses and goggles

## 🚀 Quick Start

### **1. Download Dataset**

#### **Option A: Automated Download (Recommended)**
```bash
# Get your Roboflow API key from: https://app.roboflow.com/settings/api
python integrate_roboflow.py
```

#### **Option B: Manual Download**
1. Go to: https://app.roboflow.com/kartick025/construction-safety/1
2. Click "Download" → "YOLOv8"
3. Extract the zip file
4. Copy contents to `datasets/ppe-balanced/`

### **2. Train the Model**
```bash
python train_construction_safety.py
```

### **3. Start the API**
```bash
python start_api.py
```

### **4. Access the System**
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📊 Expected Performance

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

### **Augmentation Parameters**
```yaml
HSV-Hue: 0.015
HSV-Saturation: 0.7
HSV-Value: 0.4
Rotation: ±15°
Translation: 0.1
Scale: 0.5-1.5
Horizontal Flip: 50%
Mosaic: 100%
```

### **Expected Results**
- **mAP50**: >0.85+
- **mAP50-95**: >0.65+
- **Inference Speed**: ~50ms per image (GPU)
- **Accuracy**: >95% for clear construction site images

## 🔧 API Endpoints

### **POST /detect**
Upload an image for PPE detection
```python
import requests

files = {'file': open('construction_site.jpg', 'rb')}
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
    "helmet": 2,
    "vest": 2,
    "gloves": 1,
    "glasses": 1
  },
  "compliance_status": "NON-COMPLIANT",
  "missing_ppe": ["gloves"],
  "annotated_image": "/static/detections/construction_site.jpg"
}
```

### **POST /batch-detect**
Process multiple construction site images
```python
files = [
    ('files', open('site1.jpg', 'rb')),
    ('files', open('site2.jpg', 'rb'))
]
response = requests.post('http://localhost:8000/batch-detect', files=files)
```

## 🏗️ Construction Site Use Cases

### **Real-time Safety Monitoring**
- **Live Detection**: Monitor workers in real-time
- **Compliance Checking**: Automatic safety compliance verification
- **Alert System**: Immediate notifications for non-compliance

### **Safety Reporting**
- **Automated Reports**: Generate daily safety compliance reports
- **Trend Analysis**: Track safety compliance over time
- **Incident Documentation**: Record safety violations with timestamps

### **Training & Education**
- **Safety Training**: Visual demonstrations of proper PPE usage
- **Compliance Education**: Show workers what's required
- **Best Practices**: Highlight correct safety equipment usage

## 📁 Project Structure

```
construction-safety-ppe/
├── datasets/
│   └── ppe-balanced/              # Construction Safety Dataset
│       ├── train/
│       │   ├── images/           # Training images
│       │   └── labels/           # Training labels
│       ├── valid/
│       │   ├── images/           # Validation images
│       │   └── labels/           # Validation labels
│       └── test/
│           ├── images/           # Test images
│           └── labels/           # Test labels
├── app.py                         # FastAPI backend
├── train_construction_safety.py   # Training script
├── integrate_roboflow.py         # Dataset download
├── start_api.py                  # API startup
├── data.yaml                     # Dataset configuration
├── templates/index.html          # Web interface
└── static/                       # Static files
```

## 🎯 Construction Site Specific Features

### **Enhanced Detection for Construction Environments**
- **Dust & Debris**: Optimized for dusty construction environments
- **Lighting Conditions**: Works in various lighting conditions
- **Multiple Workers**: Detects PPE on multiple workers simultaneously
- **Partial Occlusion**: Handles partially obscured PPE items

### **Safety Compliance Features**
- **Real-time Monitoring**: Continuous safety compliance checking
- **Violation Alerts**: Immediate notifications for safety violations
- **Compliance Reports**: Automated safety compliance reporting
- **Historical Tracking**: Track safety compliance over time

## 🔧 Customization for Construction Sites

### **Adjusting Detection Sensitivity**
```python
# In app.py, modify for construction site conditions
results = model.predict(
    image,
    conf=0.25,      # Lower = more sensitive (good for dusty conditions)
    iou=0.45,       # Lower = more detections
    imgsz=960,      # Higher = better accuracy for small PPE
    max_det=300     # Maximum detections per image
)
```

### **Construction Site Specific Augmentations**
```python
# In train_construction_safety.py
augmentation_config = {
    'hsv_h': 0.015,     # Handle different lighting
    'hsv_s': 0.7,       # Enhance color visibility
    'hsv_v': 0.4,       # Handle shadows and bright spots
    'degrees': 15.0,    # Worker movement variations
    'translate': 0.1,   # Position variations
    'scale': 0.5,       # Distance variations
    'fliplr': 0.5,      # Left/right handed workers
    'mosaic': 1.0,       # Multiple workers in one image
}
```

## 📈 Performance Optimization

### **GPU Acceleration for Construction Sites**
```bash
# Install CUDA for faster processing
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### **Batch Processing for Multiple Sites**
```python
# Process multiple construction site images
results = model.predict(images, batch=8)
```

## 🐛 Troubleshooting

### **Common Construction Site Issues**

1. **Dusty Conditions**
   ```python
   # Increase confidence threshold
   conf=0.3  # Instead of 0.25
   ```

2. **Poor Lighting**
   ```python
   # Use image enhancement
   enhanced_image = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
   ```

3. **Multiple Workers**
   ```python
   # Increase max detections
   max_det=500  # Instead of 300
   ```

## 📚 Additional Resources

- **Roboflow Dataset**: https://app.roboflow.com/kartick025/construction-safety/1
- **Construction Safety Standards**: OSHA guidelines
- **PPE Requirements**: Industry safety standards
- **YOLOv8 Documentation**: https://docs.ultralytics.com/

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for construction site scenarios
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **kartick025** for the Construction Safety dataset
- **Roboflow** for dataset hosting and management
- **Ultralytics** for YOLOv8 implementation
- **Construction Industry** for safety standards and requirements

---

**🏗️ Ready to enhance construction site safety with AI-powered PPE detection!**
