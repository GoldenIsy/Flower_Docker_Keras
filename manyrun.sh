#!/bin/bash

server_name="server"

docker run --rm --network flower -p 8085:8085 --name ${server_name} -v 'C:\Users\Manue\Desktop\docker\fishdata.pkl:/app' flwr_server_keras &
sleep 15 # Sleep for 2s to give the server enough time to start

clients=3


rounds=${clients--}

for i in `seq 1 ${rounds}`; do
    echo "Starting client $i"
    docker run --rm --network flower --name client${i} -v 'C:\Users\Manue\Desktop\docker\fishdata.pkl:/app' flwr_client_keras --partition=${i} --clients=${clients} --grpc_server_address=${server_name} &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait

$SHELL