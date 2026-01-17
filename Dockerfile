FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY preprocess.py .
COPY train_and_save_preds.py .
COPY viz_light.py .
COPY README.md .

CMD ["python", "train_and_save_preds.py"]