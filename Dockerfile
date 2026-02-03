FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install && pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY train.py api.py  tests/ ./

RUN pytest -v 

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]




