# Download Pretrained Models & Datasets for PPE Detection

## 🚀 **FASTEST OPTION: Transfer Learning** ⭐ (RECOMMENDED)

Instead of training from scratch (20+ hours), use **transfer learning** (1-2 hours)!

### **Quick Start:**
```bash
python use_pretrained_model.py
```

**What it does:**
- Downloads YOLO model pretrained on COCO (80 classes, 118K images)
- Fine-tunes on YOUR PPE dataset for just 10 epochs
- **Time:** 1-2 hours (vs 20+ hours from scratch)
- **Accuracy:** 65-75% mAP@0.5 (close to full training!)

---

## 📥 **Option 1: Pretrained YOLO Models** (Recommended)

### **Auto-Download from Ultralytics** (Built-in)

YOLOv8 models (pretrained on COCO dataset):

| Model | Size | Speed | Download Command |
|-------|------|-------|------------------|
| YOLOv8n | 6MB | Fastest | `model = YOLO('yolov8n.pt')` ✅ |
| YOLOv8s | 22MB | Fast | `model = YOLO('yolov8s.pt')` |
| YOLOv8m | 52MB | Medium | `model = YOLO('yolov8m.pt')` |
| YOLOv8l | 88MB | Slow | `model = YOLO('yolov8l.pt')` |

**They auto-download when you first use them!**

```python
from ultralytics import YOLO

# Downloads automatically if not present
model = YOLO('yolov8n.pt')

# Use directly OR fine-tune on your data
results = model('image.jpg')  # Use as-is
# OR
model.train(data='ppe_detection_dataset.yaml', epochs=10)  # Fine-tune
```

---

## 📦 **Option 2: Download Community PPE Models**

### **A) Roboflow Universe** (Best for PPE)

**URL:** https://universe.roboflow.com/

**Search for:**
- "PPE detection"
- "Hard hat detection"
- "Safety equipment"
- "Construction safety"

**Popular PPE Datasets/Models:**

1. **Construction Site Safety**
   - URL: https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety
   - Classes: Hard hat, safety vest, person
   - Download: YOLO format available

2. **PPE Detection**
   - URL: https://universe.roboflow.com/ppe-detection
   - Multiple PPE items
   - Pretrained models available

3. **Hard Hat Detection**
   - URL: https://universe.roboflow.com/hard-hat-detection
   - Specialized for helmets
   - High accuracy models

**How to use:**
1. Go to Roboflow Universe
2. Search "PPE detection"
3. Download in "YOLOv8" format
4. Use the provided model.pt file

---

### **B) Ultralytics HUB** (Official)

**URL:** https://hub.ultralytics.com/

**Steps:**
1. Create free account
2. Browse public models
3. Search: "PPE", "safety", "helmet"
4. Download .pt files
5. Use with: `model = YOLO('downloaded_model.pt')`

---

### **C) GitHub Repositories**

**Popular PPE Detection Repos:**

1. **PPE-Detection-YOLO-Deep_SORT**
   ```bash
   git clone https://github.com/AnshulSood11/PPE-Detection-YOLOv3-Deep_SORT.git
   # Contains pretrained weights for PPE
   ```

2. **Safety-Helmet-Wearing-Dataset**
   ```bash
   git clone https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset.git
   # 7,581 images with helmet annotations
   ```

3. **hardhat-detection**
   - URL: https://github.com/michailtam/hardhat-detection
   - Pretrained YOLO models included

---

## 📊 **Option 3: Download Additional Datasets**

### **To Improve Your Current Model:**

#### **1. Kaggle Datasets**

**Hard Hat Detection Dataset**
```bash
# Install kaggle
pip install kaggle

# Download (need Kaggle API key)
kaggle datasets download -d andrewmvd/hard-hat-detection
```

**Construction Site Safety**
```bash
kaggle datasets download -d snehilsanyal/construction-site-safety-image-dataset-roboflow
```

#### **2. Open Images Dataset**

**Download PPE-related images:**
```bash
pip install fiftyone

python -c "
import fiftyone.zoo as foz

# Download Open Images subset with PPE-related classes
dataset = foz.load_zoo_dataset(
    'open-images-v7',
    split='train',
    label_types=['detections'],
    classes=['Helmet', 'Glove', 'Glasses'],
    max_samples=1000
)
"
```

---

## ⚡ **RECOMMENDED APPROACH** (Transfer Learning)

### **Why Transfer Learning is Best:**

| Method | Time | Accuracy | Effort |
|--------|------|----------|--------|
| **Train from scratch** | 20-30h | 75-85% | High |
| **Transfer learning** ⭐ | 1-2h | 65-75% | Low |
| **Pretrained only** | 0h | 40-50% | None |

### **Transfer Learning Advantages:**

✅ **90% faster** than training from scratch
✅ **Uses pretrained knowledge** from 118K COCO images
✅ **Good accuracy** with minimal training
✅ **Less prone to overfitting**
✅ **Works great with small datasets**

---

## 🚀 **QUICK START GUIDE**

### **Method 1: Transfer Learning** (RECOMMENDED - 1-2 hours)

```bash
python use_pretrained_model.py
```

This will:
1. Download YOLOv8n pretrained on COCO
2. Fine-tune on your PPE dataset (10 epochs)
3. Complete in 1-2 hours
4. Give you a working PPE detector!

### **Method 2: Download Community Model** (INSTANT)

```python
# Option A: From Roboflow
# 1. Go to: https://universe.roboflow.com/
# 2. Search: "PPE detection"
# 3. Download model.pt file
# 4. Use it:

from ultralytics import YOLO
model = YOLO('path/to/downloaded_model.pt')
results = model('your_image.jpg')
```

### **Method 3: Use Base YOLO** (INSTANT but less accurate)

```python
from ultralytics import YOLO

# Use pretrained YOLO directly (no training)
model = YOLO('yolov8n.pt')  # Auto-downloads
results = model('image.jpg')

# It won't detect specific PPE but can detect "person"
```

---

## 📋 **Download Links Summary**

### **Pretrained Models:**
- ✅ **YOLOv8**: Auto-downloads in code (no manual download needed)
- ✅ **Roboflow Universe**: https://universe.roboflow.com/
- ✅ **Ultralytics HUB**: https://hub.ultralytics.com/

### **Datasets:**
- ✅ **Kaggle - Hard Hat**: https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection
- ✅ **Kaggle - Construction Safety**: https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow
- ✅ **GitHub - Safety Helmet**: https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset

### **Code Repositories:**
- ✅ **PPE Detection YOLO**: https://github.com/AnshulSood11/PPE-Detection-YOLOv3-Deep_SORT
- ✅ **Hard Hat Detection**: https://github.com/michailtam/hardhat-detection

---

## 🎯 **Which Should You Choose?**

### **Scenario 1: Need Results Fast** (1-2 hours)
```bash
python use_pretrained_model.py
```
✅ Best choice! Uses transfer learning

### **Scenario 2: Need Results NOW** (Instant)
1. Go to https://universe.roboflow.com/
2. Search "PPE detection"
3. Download a pretrained model
4. Use it directly!

### **Scenario 3: Want Best Accuracy** (20+ hours)
```bash
python train_ppe_detection.py
```
Continue your full training

---

## 💡 **Transfer Learning Example**

```python
from ultralytics import YOLO

# Download pretrained model
model = YOLO('yolov8n.pt')  # Pretrained on COCO

# Fine-tune on your PPE data (FAST - only 10 epochs)
results = model.train(
    data='ppe_detection_dataset.yaml',
    epochs=10,          # Just 10 epochs!
    imgsz=416,
    batch=2,
    freeze=10,          # Freeze backbone (faster)
    lr0=0.001,         # Lower learning rate
    device='cuda',
    project='ppe_quick',
    name='transfer_learning'
)

# Use your fine-tuned model
trained_model = YOLO('ppe_quick/transfer_learning/weights/best.pt')
results = trained_model('worker_photo.jpg')
results[0].show()
```

**Time:** 1-2 hours
**Accuracy:** 65-75% mAP@0.5
**Perfect for:** Most PPE detection applications!

---

## 📚 **Additional Resources**

### **Datasets to Augment Yours:**

1. **SafetyHelmetWearing-Dataset (SHWD)**
   - 7,581 images with helmet labels
   - GitHub: https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset

2. **SCUT-HEAD Dataset**
   - Large scale head detection
   - Can combine with helmet detection

3. **COCO Person Detection**
   - Use YOLO's person detection
   - Combine with PPE detection

### **Pretrained Weights Repositories:**

1. **Ultralytics Official**
   - https://github.com/ultralytics/assets/releases
   - All YOLOv8 weights available

2. **Model Zoo**
   - https://github.com/ultralytics/ultralytics#models
   - Various architectures and sizes

---

## ✅ **RECOMMENDED ACTION**

**For immediate results:**

```bash
# Run transfer learning (1-2 hours)
python use_pretrained_model.py
```

**This gives you:**
- ✅ Working PPE detector in 1-2 hours
- ✅ Good accuracy (65-75%)
- ✅ Uses your existing dataset
- ✅ No need to download extra datasets
- ✅ Best balance of speed vs quality

---

## 🎯 **Summary**

**You have 3 options:**

1. **Transfer Learning** (1-2 hours) ⭐
   - Run: `python use_pretrained_model.py`
   - Best choice for you!

2. **Download Community Model** (Instant)
   - Visit: https://universe.roboflow.com/
   - Search "PPE detection"
   - Download and use

3. **Full Training** (20+ hours)
   - Run: `python train_ppe_detection.py`
   - Best accuracy but slowest

**Recommendation: Use transfer learning!** It's the perfect balance. 🚀




