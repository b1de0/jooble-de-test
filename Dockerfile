FROM python:3.7

COPY requirements.txt /etc/

RUN pip install --no-cache-dir -r /etc/requirements.txt

WORKDIR /src

COPY src /src

CMD ["python", "-u", "main.py"]
