# worker.py

def main(instance_name):
    while not stop_threads:
        print(f"{instance_name} is running...")
        # do work here
        # sleep or perform loop tasks
        import time
        time.sleep(2000)
