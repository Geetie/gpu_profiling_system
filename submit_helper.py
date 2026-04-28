import requests
import sys
import time

BASE = "http://10.176.37.31:8080"
ID = "23301010041"

def start_container():
    r = requests.post(f"{BASE}/start", json={"id": ID, "gpu": 0})
    print(f"START: {r.status_code} {r.text}")
    return r.json()

def finish_container():
    r = requests.post(f"{BASE}/finish", json={"id": ID})
    print(f"FINISH: {r.status_code} {r.text}")
    return r.json()

def submit_test():
    r = requests.post(f"{BASE}/submit-test", json={"id": ID, "gpu": 1})
    print(f"SUBMIT-TEST: {r.status_code} {r.text}")
    return r.json()

def check_status(output_file):
    r = requests.get(f"{BASE}/submit_status/{output_file}")
    print(f"STATUS: {r.status_code} {r.text}")
    return r.json()

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "start"
    if cmd == "start":
        start_container()
    elif cmd == "finish":
        finish_container()
    elif cmd == "submit":
        submit_test()
    elif cmd == "status":
        output_file = sys.argv[2] if len(sys.argv) > 2 else ""
        if output_file:
            check_status(output_file)
        else:
            print("Usage: python submit_helper.py status <output_file>")
    else:
        print(f"Unknown command: {cmd}")
