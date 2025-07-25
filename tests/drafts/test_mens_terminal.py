import sys
import time
import threading
from contextlib import contextmanager


# def spinner():
#     while not done:
#         for char in "|/-\\":
#             sys.stdout.write(f'\rProcessando... {char}')
#             sys.stdout.flush()
#             time.sleep(0.1)
#
# done = False
# t = threading.Thread(target=spinner)
# t.start()
#
# # Simula tarefa
# time.sleep(5)
# done = True
# t.join()
# print("\rProcesso finalizado.      ")


@contextmanager
def display_process(message):
    stop = False
    sys.stdout.write("\n\r")
    sys.stdout.flush()

    def animate_dots():
        dots = ["", ".", "..", "...", "..", ".", ""]
        while not stop:
            for d in dots:
                if stop:
                    break
                sys.stdout.write("\033[2K\r")  # limpa conteúdo da animação
                sys.stdout.write(f"\r{message}{d} ")
                sys.stdout.flush()
                time.sleep(0.5)

    t = threading.Thread(target=animate_dots)
    t.start()

    try:
        yield
    finally:
        stop = True
        t.join()
        sys.stdout.write("\033[2K\r")  # limpa conteúdo da animação
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[2K\r")
        sys.stdout.flush()


for i in range(100):
    with display_process(f"Avaliando indivíduo {i}/{100}"):
        time.sleep(5)