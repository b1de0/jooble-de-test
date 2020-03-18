FROM python:3.7

COPY requirements.txt /etc/

RUN pip install --no-cache-dir -r /etc/requirements.txt

WORKDIR /app

COPY app /app

CMD ["python", "-u", "main.py"]
