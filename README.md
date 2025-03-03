# Build the image
docker build -t ai-coder-image .

# Run the container
docker run --env-file .env --rm -it --name zerion-coder ai-coder-image