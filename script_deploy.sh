# Replace with your ECR repository URI
ECR_REPOSITORY_URI="396873176791.dkr.ecr.us-east-2.amazonaws.com/neonai/ai-coder"

docker build -t ai-coder-image .
docker tag ai-coder-image:latest $ECR_REPOSITORY_URI:latest
aws ecr get-login-password --region us-east-2 --profile zerion | docker login --username AWS --password-stdin $ECR_REPOSITORY_URI
docker push $ECR_REPOSITORY_URI:latest