FROM python:3.10-slim

WORKDIR /app


COPY requirements.txt ./

#To avoid heavy sentence transfromer cuda support, dowmload only cpu version
RUN pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu \
  -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "Bot_streamlit_version.py", "--server.port=8501", "--server.address=0.0.0.0"]