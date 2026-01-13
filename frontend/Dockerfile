FROM python:3.11-slim
WORKDIR /app

# Install dependencies
RUN pip install streamlit requests

# Copy your frontend script
COPY frontend.py .

# Streamlit runs on port 8501 by default
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]