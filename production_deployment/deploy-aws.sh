#!/bin/bash
# AWS Deployment Script for PPE Detection System

echo "Deploying PPE Detection System to AWS..."

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "AWS CLI not found. Please install AWS CLI first."
    exit 1
fi

# Deploy CloudFormation stack
aws cloudformation create-stack \
    --stack-name ppe-detection-system \
    --template-body file://aws-cloudformation.json \
    --capabilities CAPABILITY_IAM \
    --parameters ParameterKey=InstanceType,ParameterValue=t3.medium \
                 ParameterKey=KeyPairName,ParameterValue=your-key-pair

echo "Stack creation initiated. Check AWS Console for progress."
echo "Once complete, you can access your application at the public IP address."
