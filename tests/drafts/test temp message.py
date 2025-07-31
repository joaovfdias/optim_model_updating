# test_console.py
from external.additional_functions import PersistentConsoleManager
import time
from multiprocessing import freeze_support

def main():
    PersistentConsoleManager.send_message("Inicializando avaliação")
    time.sleep(2)

    PersistentConsoleManager.send_message("Executando ANSYS")
    time.sleep(2)

    PersistentConsoleManager.send_message("Otimizando indivíduo 5/10")
    time.sleep(2)

    PersistentConsoleManager.send_message("Aguardando saída...")
    time.sleep(2)

    # Fecha a janela após input
    PersistentConsoleManager.close()

if __name__ == '__main__':
    freeze_support()
    main()
