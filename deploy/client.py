# Copyright (C) 2026 MOST Contributors.
import time
import pickle
import socket
import struct

from transformers import AutoProcessor


class Client:
    def __init__(self, host="localhost", port=10086, model_path=None):
        self.host = host
        self.port = port
        self.processor = (
            AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True, use_fast=False
            )
            if model_path
            else None
        )
        self._connect_with_retry(max_retries=None, retry_interval=1)
        print(f"Client connected to server at {self.host}:{self.port}.")

    def _connect_with_retry(self, max_retries=None, retry_interval=1):
        """Connect with retry logic. max_retries=None implies infinite."""
        retry_count = 0
        while True:
            try:
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.connect((self.host, self.port))
                return
            except (ConnectionRefusedError, socket.error) as e:
                retry_count += 1
                time.sleep(retry_interval)
                if max_retries is not None and retry_count >= max_retries:
                    raise ConnectionError(
                        f"Failed to connect to {self.host}:{self.port} after {retry_count} retries: {e}"
                    ) from e

    def _send_with_length_prefix(self, data):
        serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        self.client_socket.sendall(struct.pack(">I", len(serialized)) + serialized)

    def _recv_exact(self, length):
        chunks = []
        received = 0
        while received < length:
            chunk = self.client_socket.recv(length - received)
            if not chunk:
                raise ConnectionError("Connection closed while receiving a response.")
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    def _recv_with_length_prefix(self):
        len_data = self._recv_exact(4)
        data_len = struct.unpack(">I", len_data)[0]
        return pickle.loads(self._recv_exact(data_len))

    def __call__(self, **data):
        robot_type = data.get("task_id")
        self._send_with_length_prefix(data)
        actions = self._recv_with_length_prefix()
        action = self.processor.decode_action(actions, robot_type=robot_type)
        return action

    def close(self):
        self.client_socket.close()
        print("Client connection closed.")
