#!/usr/bin/env python3
"""
Production Deployment Script for PPE Detection System
Automated deployment to cloud platforms
"""

import os
import json
import subprocess
import shutil
from pathlib import Path
import yaml

class ProductionDeployer:
    def __init__(self, project_name="ppe-detection-system"):
        """Initialize production deployment"""
        self.project_name = project_name
        self.deployment_dir = Path("production_deployment")
        self.deployment_dir.mkdir(exist_ok=True)
        
        print(f"[INFO] Production deployment initialized for: {project_name}")
    
    def create_docker_setup(self):
        """Create Docker configuration for production"""
        print("\n[INFO] Creating Docker configuration...")
        
        # Create Dockerfile
        dockerfile_content = """FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p static/detections compliance_data/reports compliance_data/charts

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=ppe_web_app.py
ENV FLASK_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "ppe_web_app:app"]
"""
        
        with open(self.deployment_dir / "Dockerfile", 'w') as f:
            f.write(dockerfile_content)
        
        # Create docker-compose.yml
        docker_compose_content = """version: '3.8'

services:
  ppe-detection:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - MODEL_PATH=ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt
    volumes:
      - ./compliance_data:/app/compliance_data
      - ./static:/app/static
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - ppe-detection
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

volumes:
  redis_data:
"""
        
        with open(self.deployment_dir / "docker-compose.yml", 'w') as f:
            f.write(docker_compose_content)
        
        # Create nginx configuration
        nginx_config = """events {
    worker_connections 1024;
}

http {
    upstream ppe_app {
        server ppe-detection:5000;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        location / {
            proxy_pass http://ppe_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /health {
            proxy_pass http://ppe_app/health;
            access_log off;
        }
    }
}
"""
        
        with open(self.deployment_dir / "nginx.conf", 'w') as f:
            f.write(nginx_config)
        
        print("[SUCCESS] Docker configuration created")
    
    def create_aws_deployment(self):
        """Create AWS deployment configuration"""
        print("\n[INFO] Creating AWS deployment configuration...")
        
        # Create AWS CloudFormation template
        cloudformation_template = {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": "PPE Detection System on AWS",
            "Parameters": {
                "InstanceType": {
                    "Type": "String",
                    "Default": "t3.medium",
                    "Description": "EC2 instance type"
                },
                "KeyPairName": {
                    "Type": "String",
                    "Description": "EC2 Key Pair name"
                }
            },
            "Resources": {
                "PPEDetectionInstance": {
                    "Type": "AWS::EC2::Instance",
                    "Properties": {
                        "ImageId": "ami-0c02fb55956c7d316",  # Amazon Linux 2
                        "InstanceType": {"Ref": "InstanceType"},
                        "KeyName": {"Ref": "KeyPairName"},
                        "SecurityGroups": [{"Ref": "PPEDetectionSecurityGroup"}],
                        "UserData": {
                            "Fn::Base64": {
                                "Fn::Sub": """
#!/bin/bash
yum update -y
yum install -y docker git
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Clone and deploy application
git clone https://github.com/your-repo/ppe-detection-system.git
cd ppe-detection-system
docker-compose up -d
"""
                            }
                        }
                    }
                },
                "PPEDetectionSecurityGroup": {
                    "Type": "AWS::EC2::SecurityGroup",
                    "Properties": {
                        "GroupDescription": "Security group for PPE Detection System",
                        "SecurityGroupIngress": [
                            {
                                "IpProtocol": "tcp",
                                "FromPort": 80,
                                "ToPort": 80,
                                "CidrIp": "0.0.0.0/0"
                            },
                            {
                                "IpProtocol": "tcp",
                                "FromPort": 443,
                                "ToPort": 443,
                                "CidrIp": "0.0.0.0/0"
                            },
                            {
                                "IpProtocol": "tcp",
                                "FromPort": 22,
                                "ToPort": 22,
                                "CidrIp": "0.0.0.0/0"
                            }
                        ]
                    }
                }
            },
            "Outputs": {
                "InstanceId": {
                    "Description": "Instance ID of the PPE Detection System",
                    "Value": {"Ref": "PPEDetectionInstance"}
                },
                "PublicIP": {
                    "Description": "Public IP address of the PPE Detection System",
                    "Value": {"Fn::GetAtt": ["PPEDetectionInstance", "PublicIp"]}
                }
            }
        }
        
        with open(self.deployment_dir / "aws-cloudformation.json", 'w') as f:
            json.dump(cloudformation_template, f, indent=2)
        
        # Create deployment script
        aws_deploy_script = """#!/bin/bash
# AWS Deployment Script for PPE Detection System

echo "Deploying PPE Detection System to AWS..."

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "AWS CLI not found. Please install AWS CLI first."
    exit 1
fi

# Deploy CloudFormation stack
aws cloudformation create-stack \\
    --stack-name ppe-detection-system \\
    --template-body file://aws-cloudformation.json \\
    --capabilities CAPABILITY_IAM \\
    --parameters ParameterKey=InstanceType,ParameterValue=t3.medium \\
                 ParameterKey=KeyPairName,ParameterValue=your-key-pair

echo "Stack creation initiated. Check AWS Console for progress."
echo "Once complete, you can access your application at the public IP address."
"""
        
        with open(self.deployment_dir / "deploy-aws.sh", 'w') as f:
            f.write(aws_deploy_script)
        
        os.chmod(self.deployment_dir / "deploy-aws.sh", 0o755)
        
        print("[SUCCESS] AWS deployment configuration created")
    
    def create_azure_deployment(self):
        """Create Azure deployment configuration"""
        print("\n[INFO] Creating Azure deployment configuration...")
        
        # Create Azure Resource Manager template
        arm_template = {
            "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
            "contentVersion": "1.0.0.0",
            "parameters": {
                "vmSize": {
                    "type": "string",
                    "defaultValue": "Standard_B2s",
                    "metadata": {
                        "description": "Size of the virtual machine"
                    }
                }
            },
            "variables": {
                "vmName": "ppe-detection-vm",
                "nicName": "ppe-detection-nic",
                "publicIPAddressName": "ppe-detection-ip",
                "subnetName": "ppe-detection-subnet",
                "vnetName": "ppe-detection-vnet"
            },
            "resources": [
                {
                    "type": "Microsoft.Network/publicIPAddresses",
                    "apiVersion": "2021-05-01",
                    "name": "[variables('publicIPAddressName')]",
                    "location": "[resourceGroup().location]",
                    "properties": {
                        "publicIPAllocationMethod": "Dynamic"
                    }
                },
                {
                    "type": "Microsoft.Network/virtualNetworks",
                    "apiVersion": "2021-05-01",
                    "name": "[variables('vnetName')]",
                    "location": "[resourceGroup().location]",
                    "properties": {
                        "addressSpace": {
                            "addressPrefixes": ["10.0.0.0/16"]
                        },
                        "subnets": [
                            {
                                "name": "[variables('subnetName')]",
                                "properties": {
                                    "addressPrefix": "10.0.0.0/24"
                                }
                            }
                        ]
                    }
                },
                {
                    "type": "Microsoft.Network/networkInterfaces",
                    "apiVersion": "2021-05-01",
                    "name": "[variables('nicName')]",
                    "location": "[resourceGroup().location]",
                    "dependsOn": [
                        "[resourceId('Microsoft.Network/publicIPAddresses', variables('publicIPAddressName'))]",
                        "[resourceId('Microsoft.Network/virtualNetworks', variables('vnetName'))]"
                    ],
                    "properties": {
                        "ipConfigurations": [
                            {
                                "name": "ipconfig1",
                                "properties": {
                                    "subnet": {
                                        "id": "[resourceId('Microsoft.Network/virtualNetworks/subnets', variables('vnetName'), variables('subnetName'))]"
                                    },
                                    "publicIPAddress": {
                                        "id": "[resourceId('Microsoft.Network/publicIPAddresses', variables('publicIPAddressName'))]"
                                    }
                                }
                            }
                        ]
                    }
                },
                {
                    "type": "Microsoft.Compute/virtualMachines",
                    "apiVersion": "2021-11-01",
                    "name": "[variables('vmName')]",
                    "location": "[resourceGroup().location]",
                    "dependsOn": [
                        "[resourceId('Microsoft.Network/networkInterfaces', variables('nicName'))]"
                    ],
                    "properties": {
                        "hardwareProfile": {
                            "vmSize": "[parameters('vmSize')]"
                        },
                        "storageProfile": {
                            "imageReference": {
                                "publisher": "Canonical",
                                "offer": "UbuntuServer",
                                "sku": "18.04-LTS",
                                "version": "latest"
                            },
                            "osDisk": {
                                "createOption": "FromImage"
                            }
                        },
                        "osProfile": {
                            "computerName": "[variables('vmName')]",
                            "adminUsername": "azureuser",
                            "linuxConfiguration": {
                                "disablePasswordAuthentication": True,
                                "ssh": {
                                    "publicKeys": [
                                        {
                                            "path": "/home/azureuser/.ssh/authorized_keys",
                                            "keyData": "your-ssh-public-key"
                                        }
                                    ]
                                }
                            }
                        },
                        "networkProfile": {
                            "networkInterfaces": [
                                {
                                    "id": "[resourceId('Microsoft.Network/networkInterfaces', variables('nicName'))]"
                                }
                            ]
                        }
                    }
                }
            ]
        }
        
        with open(self.deployment_dir / "azure-template.json", 'w') as f:
            json.dump(arm_template, f, indent=2)
        
        print("[SUCCESS] Azure deployment configuration created")
    
    def create_kubernetes_deployment(self):
        """Create Kubernetes deployment configuration"""
        print("\n[INFO] Creating Kubernetes deployment configuration...")
        
        # Create Kubernetes deployment YAML
        k8s_deployment = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: ppe-detection
  labels:
    app: ppe-detection
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ppe-detection
  template:
    metadata:
      labels:
        app: ppe-detection
    spec:
      containers:
      - name: ppe-detection
        image: ppe-detection:latest
        ports:
        - containerPort: 5000
        env:
        - name: FLASK_ENV
          value: "production"
        - name: MODEL_PATH
          value: "ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ppe-detection-service
spec:
  selector:
    app: ppe-detection
  ports:
    - protocol: TCP
      port: 80
      targetPort: 5000
  type: LoadBalancer
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ppe-detection-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: ppe-detection.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ppe-detection-service
            port:
              number: 80
"""
        
        with open(self.deployment_dir / "k8s-deployment.yaml", 'w') as f:
            f.write(k8s_deployment)
        
        print("[SUCCESS] Kubernetes deployment configuration created")
    
    def create_monitoring_setup(self):
        """Create monitoring and logging setup"""
        print("\n[INFO] Creating monitoring configuration...")
        
        # Create Prometheus configuration
        prometheus_config = """global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ppe-detection'
    static_configs:
      - targets: ['ppe-detection:5000']
    metrics_path: '/metrics'
    scrape_interval: 5s
"""
        
        with open(self.deployment_dir / "prometheus.yml", 'w') as f:
            f.write(prometheus_config)
        
        # Create monitoring docker-compose
        monitoring_compose = """version: '3.8'

services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.15.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"

  kibana:
    image: docker.elastic.co/kibana/kibana:7.15.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200

volumes:
  grafana-storage:
"""
        
        with open(self.deployment_dir / "monitoring-docker-compose.yml", 'w') as f:
            f.write(monitoring_compose)
        
        print("[SUCCESS] Monitoring configuration created")
    
    def create_requirements(self):
        """Create production requirements file"""
        requirements = """Flask==2.3.3
gunicorn==21.2.0
ultralytics==8.3.214
opencv-python==4.8.1.78
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
Pillow==10.0.0
redis==4.6.0
celery==5.3.1
prometheus-client==0.17.1
"""
        
        with open(self.deployment_dir / "requirements.txt", 'w') as f:
            f.write(requirements)
        
        print("[SUCCESS] Production requirements created")
    
    def create_deployment_scripts(self):
        """Create deployment automation scripts"""
        print("\n[INFO] Creating deployment scripts...")
        
        # Local deployment script
        local_deploy = """#!/bin/bash
# Local Production Deployment Script

echo "Deploying PPE Detection System locally..."

# Build Docker image
docker build -t ppe-detection:latest .

# Stop existing containers
docker-compose down

# Start services
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 30

# Check health
curl -f http://localhost:5000/health || echo "Health check failed"

echo "Deployment complete!"
echo "Access your application at: http://localhost"
echo "Monitoring dashboard at: http://localhost:3000"
"""
        
        with open(self.deployment_dir / "deploy-local.sh", 'w') as f:
            f.write(local_deploy)
        
        os.chmod(self.deployment_dir / "deploy-local.sh", 0o755)
        
        # Production deployment script
        prod_deploy = """#!/bin/bash
# Production Deployment Script

echo "Deploying PPE Detection System to production..."

# Set environment variables
export FLASK_ENV=production
export MODEL_PATH=ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt

# Install dependencies
pip install -r requirements.txt

# Start with Gunicorn
gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 120 ppe_web_app:app

echo "Production deployment complete!"
"""
        
        with open(self.deployment_dir / "deploy-production.sh", 'w') as f:
            f.write(prod_deploy)
        
        os.chmod(self.deployment_dir / "deploy-production.sh", 0o755)
        
        print("[SUCCESS] Deployment scripts created")
    
    def copy_application_files(self):
        """Copy application files to deployment directory"""
        print("\n[INFO] Copying application files...")
        
        # Files to copy
        files_to_copy = [
            "ppe_web_app.py",
            "ppe_detection_inference.py",
            "real_time_monitoring.py",
            "compliance_reporting.py",
            "templates/",
            "static/",
            "ppe_quick_finetune/",
            "ppe_detection_dataset.yaml"
        ]
        
        for file_path in files_to_copy:
            src = Path(file_path)
            dst = self.deployment_dir / file_path
            
            if src.is_file():
                shutil.copy2(src, dst)
                print(f"Copied file: {file_path}")
            elif src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                print(f"Copied directory: {file_path}")
            else:
                print(f"Warning: {file_path} not found")
        
        print("[SUCCESS] Application files copied")
    
    def generate_deployment_guide(self):
        """Generate comprehensive deployment guide"""
        guide_content = """# 🚀 PPE Detection System - Production Deployment Guide

## 📋 Prerequisites

### System Requirements
- Docker and Docker Compose
- Python 3.9+ (for local development)
- 4GB+ RAM
- 10GB+ disk space
- GPU (optional, for faster inference)

### Cloud Requirements
- AWS Account (for AWS deployment)
- Azure Account (for Azure deployment)
- Kubernetes cluster (for K8s deployment)

## 🐳 Docker Deployment (Recommended)

### Quick Start
```bash
# Clone the repository
git clone <your-repo-url>
cd ppe-detection-system

# Build and deploy
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Access Points
- **Application**: http://localhost
- **Monitoring**: http://localhost:3000 (Grafana)
- **Metrics**: http://localhost:9090 (Prometheus)

## ☁️ Cloud Deployment

### AWS Deployment
```bash
# Deploy to AWS
./deploy-aws.sh

# Access your application
# Use the public IP from CloudFormation output
```

### Azure Deployment
```bash
# Deploy to Azure
az group create --name ppe-detection-rg --location eastus
az deployment group create --resource-group ppe-detection-rg --template-file azure-template.json
```

### Kubernetes Deployment
```bash
# Deploy to Kubernetes
kubectl apply -f k8s-deployment.yaml

# Check status
kubectl get pods
kubectl get services
```

## 📊 Monitoring Setup

### Prometheus + Grafana
```bash
# Start monitoring stack
docker-compose -f monitoring-docker-compose.yml up -d
```

### Key Metrics
- Request rate
- Response time
- Error rate
- Model inference time
- Compliance rate

## 🔧 Configuration

### Environment Variables
```bash
export FLASK_ENV=production
export MODEL_PATH=ppe_quick_finetune/yolov8n_ppe_20epochs/weights/best.pt
export REDIS_URL=redis://localhost:6379
```

### Model Configuration
- Update model path in environment variables
- Ensure model files are accessible
- Configure confidence thresholds

## 🚨 Troubleshooting

### Common Issues
1. **Model not found**: Check MODEL_PATH environment variable
2. **CUDA errors**: Use CPU-only mode or check GPU drivers
3. **Memory issues**: Increase container memory limits
4. **Port conflicts**: Change port mappings in docker-compose.yml

### Health Checks
```bash
# Check application health
curl http://localhost:5000/health

# Check container status
docker-compose ps

# View application logs
docker-compose logs ppe-detection
```

## 📈 Performance Optimization

### Production Settings
- Use Gunicorn with multiple workers
- Enable Redis for caching
- Use Nginx for load balancing
- Configure proper resource limits

### Scaling
- Horizontal scaling with multiple containers
- Load balancer configuration
- Database optimization
- CDN for static assets

## 🔒 Security

### Production Security
- Use HTTPS with SSL certificates
- Configure firewall rules
- Enable authentication
- Regular security updates
- Data encryption at rest

### Best Practices
- Regular backups
- Monitoring and alerting
- Access control
- Audit logging

## 📞 Support

### Documentation
- API documentation: http://localhost/docs
- Model documentation: README.md
- Deployment guide: DEPLOYMENT_GUIDE.md

### Contact
- Technical support: support@yourcompany.com
- Documentation: docs@yourcompany.com
- Issues: GitHub Issues

---

**Your PPE Detection System is now ready for production!** 🎉
"""
        
        with open(self.deployment_dir / "DEPLOYMENT_GUIDE.md", 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        print("[SUCCESS] Deployment guide created")
    
    def deploy_all(self):
        """Deploy complete production setup"""
        print("="*70)
        print("PRODUCTION DEPLOYMENT SETUP")
        print("="*70)
        
        # Create all deployment configurations
        self.create_docker_setup()
        self.create_aws_deployment()
        self.create_azure_deployment()
        self.create_kubernetes_deployment()
        self.create_monitoring_setup()
        self.create_requirements()
        self.create_deployment_scripts()
        self.copy_application_files()
        self.generate_deployment_guide()
        
        print("\n" + "="*70)
        print("DEPLOYMENT SETUP COMPLETE!")
        print("="*70)
        print(f"\n[SUCCESS] All deployment files created in: {self.deployment_dir}")
        print("\n[INFO] Available deployment options:")
        print("  1. Docker (Local/Cloud): docker-compose up -d")
        print("  2. AWS: ./deploy-aws.sh")
        print("  3. Azure: az deployment group create ...")
        print("  4. Kubernetes: kubectl apply -f k8s-deployment.yaml")
        print("\n[INFO] Next steps:")
        print("  1. Choose your deployment method")
        print("  2. Follow the DEPLOYMENT_GUIDE.md")
        print("  3. Configure monitoring")
        print("  4. Test your deployment")
        print("\n[SUCCESS] Your PPE Detection System is ready for production! 🚀")

def main():
    """Main deployment function"""
    deployer = ProductionDeployer()
    deployer.deploy_all()

if __name__ == "__main__":
    main()
