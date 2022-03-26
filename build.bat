#!/bin/bash
#set -e

::docker build -t flwr_client_keras --build-arg cid=cid partition=partition clients=clients grpc_server_address=grpc_server_address .



docker build -t flwr_client_keras .