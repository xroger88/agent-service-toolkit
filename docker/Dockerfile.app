FROM python:3.12.3-slim

WORKDIR /app

# need C++ compiler for pip install langchain-chroma
RUN apt-get update -y
RUN apt install build-essential manpages-dev -y

COPY requirements.txt .
RUN pip install --no-cache-dir uv
RUN uv pip install --system --no-cache -r requirements.txt

COPY client/ ./client/
COPY schema/ ./schema/
COPY streamlit_app.py .

CMD ["streamlit", "run", "streamlit_app.py"]
