# PPE Detection System - Project Structure

## 📁 Organized Directory Structure

```
sanjayai/
├── 📁 src/                          # Source code
│   ├── 📁 web_app/                  # Web application
│   │   ├── 📄 main.py              # ✨ Main Flask application (NEW)
│   │   ├── 📁 legacy/              # Legacy web apps
│   │   │   ├── modern_ppe_web_app.py
│   │   │   ├── ppe_web_app.py
│   │   │   └── [other legacy apps]
│   │   └── [other web apps]
│   ├── 📁 models/                   # Model files and weights
│   │   ├── 📁 ppe_quick_finetune/   # Trained PPE models
│   │   ├── 📁 ppe_detection_project/ # Detection project files
│   │   ├── 📁 glove_detection_project/ # Glove detection
│   │   ├── 📁 ppe_safety_vest_improvement/ # Vest improvements
│   │   ├── 📁 legacy/              # Legacy model files
│   │   ├── 📄 yolov8n.pt          # YOLO nano weights
│   │   ├── 📄 yolov8s.pt          # YOLO small weights
│   │   └── [other model files]
│   ├── 📁 datasets/                 # Dataset files
│   │   ├── 📁 combined_datasets/    # Main dataset
│   │   ├── 📁 legacy/              # Legacy datasets
│   │   ├── 📄 ppe_detection_dataset.yaml
│   │   ├── 📄 glove_detection_dataset.yaml
│   │   └── [other dataset configs]
│   ├── 📁 training/                 # Training scripts
│   │   ├── 📄 train_ppe_detection.py
│   │   ├── 📄 train_ppe_FAST.py
│   │   ├── 📄 quick_train.py
│   │   ├── 📄 train_4gb_gpu.py
│   │   ├── 📄 download_*.py
│   │   └── [other training scripts]
│   ├── 📁 utils/                    # Utility scripts
│   │   ├── 📁 legacy/              # Legacy utility scripts
│   │   │   ├── evaluate_*.py
│   │   │   ├── improve_*.py
│   │   │   ├── fix_*.py
│   │   │   └── [other legacy utils]
│   │   └── [current utility scripts]
│   └── 📁 testing/                  # Test scripts
│       ├── 📄 test_ppe_detection.py
│       ├── 📄 test_web_app.py
│       ├── 📄 comprehensive_*.py
│       └── [other test files]
├── 📁 templates/                    # HTML templates
│   ├── 📁 main/                    # Main page templates
│   │   ├── 📄 index.html           # Homepage
│   │   └── 📄 rive_ppe_frontend.html # Rive-style frontend
│   ├── 📁 live/                     # Live monitoring templates
│   │   └── 📄 live.html            # Live monitoring page
│   ├── 📁 components/               # Reusable components
│   │   ├── 📄 logo.html            # Logo display
│   │   └── 📄 logo_component.html   # Logo component
│   └── 📁 legacy/                   # Legacy templates
│       └── [other legacy templates]
├── 📁 static/                       # Static assets
│   ├── 📁 css/                     # Stylesheets
│   ├── 📁 js/                      # JavaScript files
│   └── 📁 images/                  # Images and icons
├── 📁 config/                       # Configuration files
│   ├── 📄 app_config.py            # ✨ Application configuration (NEW)
│   └── 📄 requirements.txt          # Python dependencies
├── 📁 docs/                         # Documentation
│   ├── 📄 README.md                # Main documentation
│   ├── 📄 PROJECT_SUMMARY.md       # Project summary
│   ├── 📄 TRAINING_GUIDE.md        # Training guide
│   ├── 📄 DEPLOYMENT_GUIDE.md      # Deployment guide
│   ├── 📄 *.png                    # Documentation images
│   ├── 📄 *.jpg                    # Test result images
│   ├── 📄 *.json                   # Test reports
│   └── [other documentation]
├── 📁 scripts/                      # Deployment scripts
│   ├── 📄 start_app.py             # ✨ Application starter (NEW)
│   ├── 📄 Dockerfile               # Docker configuration
│   ├── 📄 docker-compose.yml      # Docker Compose
│   └── 📁 ppe_env/                 # Environment files
├── 📁 tests/                        # Test files
│   └── [test files]
├── 📁 logs/                         # Log files (created at runtime)
└── 📄 README.md                     # Main project README
```

## 🎯 Key Improvements Made

### ✅ **Organized Structure**
- **Separated by functionality**: Web apps, models, datasets, training, utils, tests
- **Legacy preservation**: Old files moved to `legacy/` folders
- **Clear hierarchy**: Logical grouping of related files

### ✅ **New Configuration System**
- **`config/app_config.py`**: Centralized configuration
- **`config/requirements.txt`**: Updated dependencies
- **Environment management**: Proper configuration handling

### ✅ **Main Application**
- **`src/web_app/main.py`**: Clean, organized main Flask app
- **`scripts/start_app.py`**: Easy application starter
- **Template organization**: Templates grouped by functionality

### ✅ **Documentation**
- **`README.md`**: Comprehensive project documentation
- **`PROJECT_STRUCTURE.md`**: This structure guide
- **Organized docs**: All documentation in `docs/` folder

### ✅ **Clean Separation**
- **Source code**: All in `src/` directory
- **Templates**: Organized by page type
- **Static assets**: Proper static file structure
- **Configuration**: Centralized config management

## 🚀 How to Use the New Structure

### Start the Application
```bash
# Easy start with the new script
python scripts/start_app.py

# Or run the main app directly
python src/web_app/main.py
```

### Install Dependencies
```bash
pip install -r config/requirements.txt
```

### Access the Application
- **Main Dashboard**: `http://localhost:5000`
- **Live Monitoring**: `http://localhost:5000/live`

## 📋 File Categories

### 🎯 **Active Files** (Currently Used)
- `src/web_app/main.py` - Main application
- `templates/main/index.html` - Homepage
- `templates/live/live.html` - Live monitoring
- `config/app_config.py` - Configuration

### 📚 **Legacy Files** (Preserved for Reference)
- `src/web_app/legacy/` - Old web applications
- `src/utils/legacy/` - Old utility scripts
- `templates/legacy/` - Old templates
- `src/models/legacy/` - Old model files

### 🧪 **Development Files**
- `src/training/` - Training scripts
- `src/utils/` - Utility functions
- `tests/` - Test files
- `scripts/` - Deployment scripts

## 🎉 Benefits of New Structure

1. **🔍 Easy Navigation**: Clear directory structure
2. **🛠️ Better Development**: Organized by functionality
3. **📚 Clear Documentation**: Comprehensive guides
4. **🚀 Simple Deployment**: Easy startup scripts
5. **🧹 Clean Codebase**: No duplicate files in root
6. **📦 Modular Design**: Separated concerns
7. **🔄 Legacy Support**: Old files preserved but organized

---

**✨ Your PPE Detection System is now properly organized and ready for development! 🛡️**
