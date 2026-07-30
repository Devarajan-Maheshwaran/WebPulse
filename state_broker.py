"""Process 3: in-memory local IPC broker for WebPulse physiological state.

The broker never writes metrics to disk. Vision publishes the latest WESAD/rPPG
state, while Gemini Live subscribes to immediate JSON-line updates.
"""

import json
import socketserver
import threading


class StateStore:
    def __init__(self):
        self.latest = {}
        self.subscribers = set()
        self.lock = threading.Lock()

    def publish(self, state):
        message = (json.dumps(state, separators=(",", ":")) + "\n").encode("utf-8")
        with self.lock:
            self.latest = dict(state)
            targets = list(self.subscribers)
        failed = []
        for client in targets:
            try:
                client.sendall(message)
            except OSError:
                failed.append(client)
        if failed:
            with self.lock:
                for client in failed:
                    self.subscribers.discard(client)

    def subscribe(self, client):
        with self.lock:
            self.subscribers.add(client)
            latest = dict(self.latest)
        if latest:
            client.sendall((json.dumps(latest, separators=(",", ":")) + "\n").encode("utf-8"))

    def unsubscribe(self, client):
        with self.lock:
            self.subscribers.discard(client)


STORE = StateStore()


class BrokerHandler(socketserver.BaseRequestHandler):
    def handle(self):
        buffer = ""
        subscribed = False
        try:
            while True:
                data = self.request.recv(4096)
                if not data:
                    break
                buffer += data.decode("utf-8")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line.strip():
                        continue
                    message = json.loads(line)
                    if message.get("type") == "publish":
                        STORE.publish(message.get("state", {}))
                    elif message.get("type") == "subscribe":
                        STORE.subscribe(self.request)
                        subscribed = True
        finally:
            if subscribed:
                STORE.unsubscribe(self.request)


class ThreadedBroker(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    with ThreadedBroker(("127.0.0.1", 5003), BrokerHandler) as server:
        print("[State Broker] Process 3 listening on 127.0.0.1:5003")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[State Broker] Stopped.")


if __name__ == "__main__":
    main()
