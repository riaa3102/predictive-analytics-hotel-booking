# Pull base image
FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /hotel_app

# Install make and clean up
RUN apt-get update \
    && apt-get install -y make \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements_linux.txt /hotel_app/
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements_linux.txt

# Copy the current directory contents
COPY src /hotel_app/src
COPY models /hotel_app/models
COPY app.py Makefile /hotel_app/

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["make", "hotel_app"]
