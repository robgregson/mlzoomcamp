
# Use a lightweight Python image
FROM python:3.13-slim

# Set working directory inside the container
WORKDIR /app

# Prevent Python from writing bytecode files
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure Python output appears immediately
ENV PYTHONUNBUFFERED=1

# Install Pipenv
RUN pip install --no-cache-dir pipenv

# Copy dependency definitions
COPY Pipfile Pipfile.lock ./

# Install locked dependencies into the container's Python environment
RUN pipenv install --system --deploy

# Copy application files
COPY . .

EXPOSE 8080

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8080"]

