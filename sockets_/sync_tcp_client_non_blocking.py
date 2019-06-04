#!/usr/bin/env python
# -*- coding: utf-8 -*-

import socket

MAX_CONNECTIONS = 20
address_to_server = ('localhost', 8688)

clients = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for i in range(MAX_CONNECTIONS)]
for i, client in enumerate(clients):
    client.connect(address_to_server)
    client.send(bytes("hello from client number " + str(i), encoding='UTF-8'))
    print('Connected to', i)

# for i in range(MAX_CONNECTIONS):
#     print("send message to", i, 'from', MAX_CONNECTIONS)
#     clients[i].send(bytes("hello from client number " + str(i), encoding='UTF-8'))

for i, client in enumerate(clients):
    data = client.recv(1024)
    print(i, str(data))