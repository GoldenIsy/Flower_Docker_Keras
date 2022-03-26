# Flower_Docker_Keras

This Projects builds on the Example:Flower Advanced Tensorflow: https://github.com/adap/flower/tree/main/examples/advanced_tensorflow

The Problem is the Connection between the Client and Server

Both Docker-Compose and the Run function should work, but are unabel to build an connection.
Due to Size of my testDataset I was not able to upload it in the moment. Mine was placed under data.
Please select provide the path to your dataset under "manyrun.sh" or in Docker-compse (replace <dataset> with path)


How to run:
After building the Docker Images in the Main folder and the Server folder (simply with the build.bat)
you can run the "manyrun.sh" (after choosing a Dataset) or use Docker-compose up.


Current Error:
  
  
Traceback (most recent call last):
  File "./client.py", line 164, in <module>
    main()
  File "./client.py", line 119, in main
    fl.client.start_numpy_client('localhost:8085', client=client)
  File "/usr/local/lib/python3.6/dist-packages/flwr/client/app.py", line 177, in start_numpy_client
    root_certificates=root_certificates,
  File "/usr/local/lib/python3.6/dist-packages/flwr/client/app.py", line 93, in start_client
    server_message = receive()
  File "/usr/local/lib/python3.6/dist-packages/flwr/client/grpc_client/connection.py", line 113, in <lambda>
    receive: Callable[[], ServerMessage] = lambda: next(server_message_iterator)
  File "/usr/local/lib/python3.6/dist-packages/grpc/_channel.py", line 426, in __next__
    return self._next()
  File "/usr/local/lib/python3.6/dist-packages/grpc/_channel.py", line 809, in _next
    raise self
grpc._channel._MultiThreadedRendezvous: <_MultiThreadedRendezvous of RPC that terminated with:
        status = StatusCode.UNAVAILABLE
        details = "failed to connect to all addresses"
        debug_error_string = "{"created":"@1648281786.253460900","description":"Failed to pick subchannel","file":"src/core/ext/filters/client_channel/client_channel.cc","file_line":3134,"referenced_errors":[{"created":"@1648281786.253460300","description":"failed to connect to all addresses","file":"src/core/lib/transport/error_utils.cc","file_line":163,"grpc_status":14}]}"
>


Thanks for the help!
