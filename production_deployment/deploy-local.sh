#!/bin/bash
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
