FROM python:3.8

WORKDIR /usr/src/app

# update pip
RUN pip3 install --upgrade pip

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY ./server.py ./
CMD [ "python", "./server.py" ]