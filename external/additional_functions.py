import numpy as np
from scipy.optimize import linear_sum_assignment # usado para pareamento

from contextlib import contextmanager
import threading
import sys
import time

from rich.live import Live
from rich.text import Text

import multiprocessing
import win32console


class SpecialFun:
    @staticmethod
    def modal_assurance_criterion(mode1, mode2):
        num = np.dot(mode1, mode2) ** 2
        denom = np.dot(mode1, mode1) * np.dot(mode2, mode2)
        return num / denom

    @staticmethod
    def mac_error(base_modes, comp_modes):
        num_modes = base_modes.shape[0]
        # parcelas de erro do MAC
        mac_error = [abs(1 - SpecialFun.modal_assurance_criterion(base_modes[i, :], comp_modes[i, :]))
                      for i in range(num_modes)]
        # soma
        return sum(mac_error)

    @staticmethod
    def norm_freq_errors(base_freq, comp_freq):
        num_modes = len(base_freq)
        # parcelas de erro normalizado das frequências
        freq_errors = [abs((base_freq[i] - comp_freq[i]) / base_freq[i]) for i in range(num_modes)]
        # soma
        return sum(freq_errors)

    @staticmethod
    def mac_matrix(base_modes, comp_modes):
        numerador = np.abs(base_modes @ comp_modes.T)**2 # [modos, modos]
        norma_base = np.sum(base_modes**2, axis=1, keepdims=True) # [modos, 1]
        norma_comp = np.sum(comp_modes**2, axis=1, keepdims=True).T # [1, modos]
        denominador = norma_base @ norma_comp
        # if np.any(np.isclose(denominador, 0)):
        #     raise RuntimeError('Denominator cannot be zero — invalid or ill-conditioned vibration mode detected.')
        return numerador / denominador

    @staticmethod
    def pair_modes_mac(comp_freq, comp_modes, base_modes): # matrizes de modos no formato [modos, nós]
        """
        Ordena os modos numéricos com base no pareamento com modos de referência
        """
        mac = SpecialFun.mac_matrix(base_modes, comp_modes)
        # índices para maximizar o MAC:
        base_index, comp_index = linear_sum_assignment(-mac)
        # reordenação dos modos e frequências numéricos:
        paired_comp_modes = comp_modes[comp_index, :]
        paired_comp_freq = comp_freq[comp_index]
        mac_paired = mac[base_index, comp_index]

        return [paired_comp_freq, paired_comp_modes, (1 - mac_paired).sum()]


class Utilities:

    @staticmethod
    @contextmanager
    def display_process_old(message, status=True): # obsoleta
        if not status:
            yield
            return

        stop = False

        def animate_dots():
            dots = ["", ".", "..", "...", "..", ".", ""]
            while not stop:
                for d in dots:
                    if stop:
                        break
                    sys.stdout.write("\033[2K\r")  # limpa conteúdo da animação
                    sys.stdout.write(f"\r{message}{d} ")  # limpa conteúdo da animação
                    sys.stdout.flush()
                    time.sleep(0.5)

        t = threading.Thread(target=animate_dots)
        t.start()

        try:
            yield
        except Exception as e:
            stop = True
            t.join()
            sys.stdout.write("\033[2K\r")  # limpa conteúdo da animação
            sys.stdout.flush()
            return e
        finally:
            stop = True
            t.join()
            sys.stdout.write("\033[2K\r")  # limpa conteúdo da animação
            sys.stdout.flush()


    @staticmethod
    @contextmanager
    def display_process_rich(message: str, status: bool = True, speed: float = 0.4):
        """Context manager to show an animated message with dots while a process runs."""
        if not status:
            yield
            return

        stop_event = threading.Event()
        text = Text(f"{message}", style="bold cyan")

        def animate():
            dots = ["", ".", "..", "..."]
            i = 0
            while not stop_event.is_set():
                Text.text = f"{message}{dots[i % len(dots)]}"
                i += 1
                time.sleep(speed)

        thread = threading.Thread(target=animate, daemon=True)
        with Live(text, refresh_per_second=10, transient=True):  # 'transient=True' clears on exit
            thread.start()
            try:
                yield
            finally:
                stop_event.set()
                thread.join()



class PersistentConsoleManager:
    """Singleton para criar uma janela de console separada com mensagens temporárias"""

    _instance = None  # Singleton

    def __init__(self):
        self._msg_queue = multiprocessing.Queue()
        self._stop_flag = multiprocessing.Event()

        self._console_proc = multiprocessing.Process(
            target=self._console_loop,
            args=(self._msg_queue, self._stop_flag)
        )
        self._console_proc.daemon = False  # mantém processo ativo
        self._console_proc.start()

    def _console_loop(self, queue, stop_flag):
        # Cria nova janela de console
        win32console.FreeConsole()
        win32console.AllocConsole()

        sys.stdout = open("CONOUT$", "w")
        sys.stderr = open("CONOUT$", "w")

        current_message = ""
        dots = ["", ".", "..", "...", "...."]
        dot_index = 0

        while not stop_flag.is_set():
            try:
                if not queue.empty():
                    current_message = queue.get_nowait()
                    dot_index = 0  # Reset do "..." animado
            except:
                pass

            if current_message:
                print(f"\r{current_message}{dots[dot_index % len(dots)]}   ", end="", flush=True)
                dot_index += 1
            time.sleep(0.4)

        # Ao finalizar, apaga a linha e exibe mensagem final
        print("\r" + " " * 80 + "\r", end="", flush=True)
        print("Processo concluído. Você pode fechar esta janela.", flush=True)
        input("\nPressione Enter para fechar...")

    @classmethod
    def send_message(cls, msg: str):
        if cls._instance is None:
            cls._instance = PersistentConsoleManager()
        cls._instance._msg_queue.put(msg)

    @classmethod
    def close(cls):
        if cls._instance:
            cls._instance._stop_flag.set()
            cls._instance._console_proc.join()
            cls._instance = None
