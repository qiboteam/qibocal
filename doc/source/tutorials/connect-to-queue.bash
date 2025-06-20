#!/bin/bash

read -rd '' getport <<EOF
import socket

s=socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
EOF

IP=$(hostname -I | cut -d ' ' -f 1)
PORT=$(python -c "$getport")

echo "Forward your local port with:"
echo "  ssh <login-node> -L 9180:$IP:$PORT"

echo
echo "Connect to:"
echo "  https://localhost:9180"

jupyter lab --ip "$IP" --port "$PORT"
# you can also test this with:
# python -m http.server -b "$IP" "$PORT"
