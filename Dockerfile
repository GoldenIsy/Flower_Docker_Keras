FROM haifengjin/autokeras
ARG partition=1
ARG clients=10
ARG grpc_server_address=server

WORKDIR /usr/src/app

# update pip
RUN pip3 install --upgrade pip

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY ./client.py ./

# Start the client
ENTRYPOINT ["python","./client.py"]


#Start
#CMD python3 client.py --partition=$partition --clients=$clients --grpc_server_address=$grpc_server_address