import psutil
import time

def kill_ansys_process():  # sem uso, testar
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.info['name'] and 'ANSYS.exe' in proc.info['name']:
                print(f"Encerramento forçado do processo {proc.info['name']} (PID {proc.pid})")
                proc.kill()
                time.sleep(0.2)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
