import pickle
import socket
import time
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn as nn

from models import *
# from models import lenet5


@contextmanager
def timer(text=''):
    """Helper for measuring runtime"""

    time_start = time.perf_counter()
    yield
    print('---{} time: {:.4f} s'.format(text, time.perf_counter()-time_start))


def send(socket, data):
    socket.send('{}#'.format(len(data)).encode())
    socket.sendall(data.encode())


def recv(socket):
    len_str = ''
    char = socket.recv(1).decode()
    while char != '#':
        len_str += char
        char = socket.recv(1).decode()
    length = int(len_str)
    view = memoryview(bytearray(length))
    next_offset = 0
    while length - next_offset > 0:
        recv_size = socket.recv_into(view[next_offset:], length - next_offset)
        next_offset += recv_size
    return view.tobytes().decode()


def send_object(socket, obj):
    msg = pickle.dumps(obj)
    socket.send('{}#'.format(len(msg)).encode())
    socket.sendall(msg)
    return len(msg)


def recv_object(socket):
    len_str = ''
    char = socket.recv(1).decode()
    while char != '#':
        len_str += char
        char = socket.recv(1).decode()
    length = int(len_str)
    view = memoryview(bytearray(length))
    next_offset = 0
    while length - next_offset > 0:
        recv_size = socket.recv_into(view[next_offset:], length - next_offset)
        next_offset += recv_size
    obj = pickle.loads(view.tobytes())
    return obj
