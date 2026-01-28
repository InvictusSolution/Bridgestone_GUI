import socket
import sys

PI_IP = "192.168.1.3"   # Raspberry Pi IP
PORT = 5000

def send(cmd):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((PI_IP, PORT))
        s.send(cmd.encode())
        s.close()
        print(f"Relay command sent: {cmd}")
    except Exception as e:
        print("Relay communication error:", e)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 master.py ON | OFF")
        sys.exit(1)

    command = sys.argv[1].upper()

    if command not in ("ON", "OFF"):
        print("Invalid command")
        sys.exit(1)

    send(command)
