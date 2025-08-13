FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
LABEL authors="YOUR_NAME"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
#COPY requirements.txt .
#RUN pip3 install -r requirements.txt

RUN cd segment-anything && pip install e .

# Copy necessary files
COPY tools tools/
COPY checkpoints checkpoints/
COPY main.py .

CMD ["python", "main.py", "-i", "/input", "-o", "/output"]
