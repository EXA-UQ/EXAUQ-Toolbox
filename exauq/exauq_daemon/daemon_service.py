import socket
import selectors


def start_service(socket_path):
    sel = selectors.DefaultSelector()

    def accept_wrapper(sock):
        conn, addr = sock.accept()
        print('Accepted connection from', addr)
        conn.setblocking(False)
        sel.register(conn, selectors.EVENT_READ, read_connection)

    def read_connection(conn, mask):
        data = conn.recv(1000)
        if data:
            print('Received:', data.decode())
            # Handle data...
        else:
            print('Closing connection')
            sel.unregister(conn)
            conn.close()

    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_sock.bind(socket_path)
    server_sock.listen()
    server_sock.setblocking(False)
    sel.register(server_sock, selectors.EVENT_READ, accept_wrapper)

    while True:
        events = sel.select(timeout=None)
        for key, mask in events:
            callback = key.data
            callback(key.fileobj, mask)
