import ssl

def serve(port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_KeywordServiceServicer_to_server(
        NltkService(), server)
    server.add_insecure_port('[::]:' + str(port))
    server.start()
    print("Listening on port {}..".format(port))
    try:
        while True:
            time.sleep(10000)
    except KeyboardInterrupt:
        server.stop(0)


if __name__== "__main__":
    serve(6000)