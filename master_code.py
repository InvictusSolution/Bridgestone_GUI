import paramiko
import socket
import time

PI_IP = "192.168.1.3"
PI_USER = "invictus"
PI_PASSWORD = "2904"
RELAY_PORT = 5000

# ---------------- SSH START SERVER ----------------
def start_relay_server():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(PI_IP, username=PI_USER, password=PI_PASSWORD)

    cmd = "python3 /home/invictus/Desktop/relay_off.py"

    ssh.exec_command(cmd)
    ssh.close()
    print("Relay server started on Raspberry Pi")

# def start_relay_service():
#     ssh = paramiko.SSHClient()
#     ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#     ssh.connect(PI_IP, username=PI_USER, password=PI_PASSWORD)
#
#     ssh.exec_command("sudo systemctl start relay.service")
#     ssh.close()
#
#     print("Relay service started")
# ---------------- SEND RELAY COMMAND ----------------
def send(cmd):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((PI_IP, RELAY_PORT))
    s.send(cmd.encode())
    s.close()
    print(f"Relay {cmd}")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    start_relay_server()     # Start Pi GPIO code

