# Use the official Python 3.9 image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy all files from the current directory to /app in the container
COPY . /app

# List directory contents to confirm files are copied
RUN ls /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the default command to run the server script
CMD ["python", "server.py"]
