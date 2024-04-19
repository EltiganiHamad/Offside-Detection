# Use the official Python 3.9 image as the base image
FROM python:3.9

# Install system dependencies (if needed)
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx



# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .


RUN mkdir -p outputs

Expose 5000

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run your application
CMD ["python", "app.py"]