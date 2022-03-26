docker network create flower

::Server

docker run --rm --network flower -p 8080:8080 --name server -v <data>\fishdata.pkl:/app flwr_server_keras


::Client
docker run -it --network flower -v <data>:/app flwr_client_keras --cid=1 --partition=0 --clients=1 --grpc_server_address=server
