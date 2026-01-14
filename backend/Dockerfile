FROM python:3.11-slim
WORKDIR /app

# Since the 'context' is ./backend, this looks for backend/requirements.txt
COPY requirements.txt . 
RUN pip install -r requirements.txt

COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]