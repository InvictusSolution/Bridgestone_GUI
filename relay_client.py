import paramiko
import socket
import sys
import time

PI_IP = "192.168.1.3"
PI_USER = "invictus"
PI_PASSWORD = "2904"
RELAY_PORT = 5000

RELAY_SERVER_CMD = "python3 /home/invictus/Desktop/relay_off.py"


# ---------------- START RELAY SERVER ----------------
def start_relay_server():
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(PI_IP, username=PI_USER, password=PI_PASSWORD)

        ssh.exec_command(RELAY_SERVER_CMD)
        ssh.close()

        print("Relay server started on Raspberry Pi")
        time.sleep(1)  # allow server to bind socket

    except Exception as e:
        print("Failed to start relay server:", e)


# ---------------- SEND RELAY COMMAND ----------------
def send_relay_command(cmd):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((PI_IP, RELAY_PORT))
        s.send(cmd.encode())
        s.close()

        print(f"Relay command sent: {cmd}")

    except Exception as e:
        print("Failed to send relay command:", e)


# ---------------- MAIN ----------------
if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python3 relay_client.py ON | OFF")
        sys.exit(1)

    command = sys.argv[1].upper()

    if command not in ("ON", "OFF"):
        print("Invalid command. Use ON or OFF.")
        sys.exit(1)

    # Start relay server (safe even if already running)
    start_relay_server()

    # Send explicit relay command
    send_relay_command(command)
