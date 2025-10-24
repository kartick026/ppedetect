# PPE Detection System

A comprehensive AI-powered Personal Protective Equipment (PPE) detection system using YOLOv8 for real-time monitoring and compliance checking.

## 🏗️ Project Structure

```
sanjayai/
├── src/                          # Source code
│   ├── web_app/                  # Web application
│   │   ├── main.py              # Main Flask application
│   │   ├── modern_ppe_web_app.py # Modern web app
│   │   └── ppe_web_app.py       # Legacy web app
│   ├── models/                   # Model files and weights
│   │   ├── ppe_quick_finetune/   # Trained PPE models
│   │   ├── ppe_detection_project/ # Detection project files
│   │   └── *.pt                 # YOLO model weights
│   ├── datasets/                 # Dataset files
│   │   ├── combined_datasets/    # Main dataset
│   │   └── *.yaml              # Dataset configurations
│   ├── training/                 # Training scripts
│   │   ├── train_*.py           # Training scripts
│   │   ├── quick_*.py           # Quick training
│   │   └── download_*.py        # Data download scripts
│   ├── utils/                    # Utility scripts
│   │   ├── evaluate_*.py        # Evaluation scripts
│   │   ├── improve_*.py         # Improvement scripts
│   │   └── fix_*.py             # Fix scripts
│   └── testing/                  # Test scripts
│       ├── test_*.py            # Test files
│       └── comprehensive_*.py    # Comprehensive tests
├── templates/                    # HTML templates
│   ├── main/                    # Main page templates
│   │   ├── index.html           # Homepage
│   │   └── rive_ppe_frontend.html # Rive-style frontend
│   ├── live/                     # Live monitoring templates
│   │   └── live.html            # Live monitoring page
│   ├── components/               # Reusable components
│   │   ├── logo.html            # Logo display
│   │   └── logo_component.html   # Logo component
│   └── legacy/                   # Legacy templates
├── static/                       # Static assets
│   ├── css/                     # Stylesheets
│   ├── js/                      # JavaScript files
│   └── images/                  # Images and icons
├── config/                       # Configuration files
│   ├── app_config.py            # Application configuration
│   └── requirements.txt          # Python dependencies
├── docs/                         # Documentation
│   ├── *.md                     # Markdown documentation
│   ├── *.png                    # Documentation images
│   └── *.json                   # Test reports
├── scripts/                      # Deployment scripts
│   ├── Dockerfile               # Docker configuration
│   ├── docker-compose.yml      # Docker Compose
│   └── ppe_env/                 # Environment files
├── tests/                        # Test files
└── logs/                         # Log files
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Webcam or camera device

### Installation

1. **Clone the repository**
```bash
   git clone <repository-url>
   cd sanjayai
```

2. **Install dependencies**
```bash
   pip install -r config/requirements.txt
```

3. **Set up configuration**
```bash
   python config/app_config.py
```

4. **Run the application**
```bash
   python src/web_app/main.py
   ```

5. **Access the web interface**
   - Open your browser and go to `http://localhost:5000`

## 🎯 Features

### Core Functionality
- **Real-time PPE Detection**: Live camera monitoring with YOLOv8
- **Compliance Checking**: Automatic safety compliance verification
- **Multi-class Detection**: Helmet, Safety Vest, Goggles, Gloves
- **People Counting**: Automatic person detection and counting
- **Web Interface**: Modern, responsive web dashboard

### Detection Classes
- **Helmet** (Primary safety equipment)
- **Safety Vest** (High-visibility clothing)
- **Goggles** (Eye protection)
- **Gloves** (Hand protection)

### Web Interface Features
- **Dashboard**: Main control panel with statistics
- **Live Monitoring**: Real-time camera feed with detection overlays
- **Image Upload**: Single image analysis and detection
- **Compliance Reports**: Detailed safety compliance reports
- **Interactive Logo**: Animated helmet logo with hover effects

## 🛠️ Configuration

### Model Configuration
Edit `config/app_config.py` to customize:
- Model paths and weights
- Detection confidence thresholds
- Camera settings
- Compliance requirements

### Web Application Settings
- Host and port configuration
- Debug mode settings
- Threading options

## 📊 Training

### Quick Training
```bash
python src/training/quick_train.py
```

### Full Training
```bash
python src/training/train_ppe_detection.py
```

### Model Evaluation
```bash
python src/utils/evaluate_model.py
```

## 🧪 Testing

### Run Tests
```bash
python -m pytest tests/
```

### Individual Test Files
```bash
python tests/test_ppe_detection.py
python tests/test_web_app.py
```

## 📚 Documentation

- **User Guide**: `docs/user_guide/`
- **API Documentation**: `docs/api/`
- **Deployment Guide**: `docs/deployment/`
- **Training Guide**: `docs/TRAINING_GUIDE.md`

## 🐳 Docker Deployment

### Build and Run
```bash
docker-compose up --build
```

### Production Deployment
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## 🔧 Development

### Code Organization
- **Web App**: `src/web_app/` - Flask application and routes
- **Models**: `src/models/` - YOLO models and weights
- **Training**: `src/training/` - Model training scripts
- **Utils**: `src/utils/` - Utility functions and helpers
- **Tests**: `tests/` - Test files and validation

### Adding New Features
1. Create feature branch
2. Add tests for new functionality
3. Update documentation
4. Submit pull request

## 📈 Performance

### Model Performance
- **mAP@0.5**: >0.70 (Helmet detection)
- **Inference Speed**: 30-50 FPS (GTX 1650)
- **Memory Usage**: <4GB VRAM
- **Accuracy**: 95%+ for helmet detection

### System Requirements
- **Minimum**: 4GB RAM, CPU-only inference
- **Recommended**: 8GB RAM, CUDA GPU
- **Optimal**: 16GB RAM, RTX 3060+ GPU

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics YOLOv8**: Core detection model
- **OpenCV**: Computer vision processing
- **Flask**: Web framework
- **Bootstrap**: Frontend styling

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation in `docs/`
- Review test files for usage examples

---

**PPE Detection System** - Ensuring workplace safety through AI-powered monitoring 🛡️