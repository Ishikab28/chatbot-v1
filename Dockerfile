FROM python:3.10

EXPOSE 8080
WORKDIR /app

COPY . ./

RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "1.py", "u1.py", "--server.port=8080", "--server.address=0.0.0.0"]
