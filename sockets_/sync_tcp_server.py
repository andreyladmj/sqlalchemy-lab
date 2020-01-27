import socket

# Задаем адрес сервера
SERVER_ADDRESS = ('', 8688)

# Настраиваем сокет
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(SERVER_ADDRESS)
server_socket.listen(10)
print('server is running, please, press ctrl+c to stop')

# Слушаем запросы
while True:
    connection, address = server_socket.accept()
    # connection.setblocking(1)
    print("new connection from {address}".format(address=address))

    # while True:
    #     try:
    #         print('recv')
    #         data = connection.recv(1024)
    #         print(str(data))
    #         if not data: break
    #     except BlockingIOError:
    #         pass
    data = connection.recv(1024)
    print(str(data))

    connection.send(bytes('Hello from server!', encoding='UTF-8'))

    connection.close()