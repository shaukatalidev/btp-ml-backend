### Dockerfile

Below is the `Dockerfile` used for building the application container:

```dockerfile
# Use the official lightweight Python image
FROM python:3.11.5-slim

# Set environment variables to avoid writing .pyc files and to ensure unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y \
  ffmpeg \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1 \
  && rm -rf /var/lib/apt/lists/*

# Create a directory for the application code
RUN mkdir /code

# Set the working directory inside the container
WORKDIR /code

# Copy only the requirements file first to leverage Docker caching
COPY requirements.txt .

# Install the dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the application runs on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]
```

### Running the Dockerfile

To build and run the Docker container, use the following commands:

```bash
# Build the Docker image
docker build -t my-fastapi-app .

# Run the Docker container
docker run -p 9000:9000 my-fastapi-app
```

### Accessing the API Documentation

Once the container is running, you can access the automatically generated API documentation at the following URL:

```
http://localhost:9000/docs
```
