import sys
import time
import threading
import random
import queue

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


NUM_ITERATIONS = 50
SLEEP_TIME = 0.5


def run_optimization_simulation(result_queue: queue.Queue, num_iterations=NUM_ITERATIONS, sleep_time=SLEEP_TIME):
    try:
        for i in range(num_iterations):
            fitness = random.uniform(0.1, 1.0) / (i + 1)
            result_queue.put((i, fitness))
            print(f"[OPTIMIZER] Iteration {i}: fitness = {fitness:.5f}")
            time.sleep(sleep_time)
    except Exception as e:
        print(f"[OPTIMIZER] Error: {e}")
    finally:
        result_queue.put(("DONE", None))


class PlotWindow(QMainWindow):
    def __init__(self, result_queue: queue.Queue):
        super().__init__()
        self.setWindowTitle("Live Fitness Plot")
        self.canvas = FigureCanvas(Figure(figsize=(6, 4)))
        self.ax = self.canvas.figure.subplots()
        self.ax.set_title("Fitness Evolution")
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Fitness")
        self.line, = self.ax.plot([], [], '-o', label="Fitness")
        self.ax.legend()

        # status label (mensagem temporária)
        self.status_label = QLabel("Starting...", self)
        self.status_label.setStyleSheet("font-weight: bold;")

        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.canvas)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.x_data = []
        self.y_data = []
        self.last_iteration = -1
        self.result_queue = result_queue
        self.done = False

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(200)  # refresh faster

    def update_plot(self):
        updated = False
        while not self.result_queue.empty():
            item = self.result_queue.get()
            if item[0] == "DONE":
                self.done = True
                self.status_label.setText("Optimization completed.")
            else:
                i, fitness = item
                self.x_data.append(i)
                self.y_data.append(fitness)
                self.status_label.setText(f"Avaliando indivíduo {i+1}/{NUM_ITERATIONS}...")
                updated = True

        if updated:
            self.line.set_data(self.x_data, self.y_data)
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw()

        if self.done:
            self.timer.stop()


def main():
    result_queue = queue.Queue()
    optimizer_thread = threading.Thread(target=run_optimization_simulation,
                                        args=(result_queue,),
                                        daemon=True)
    optimizer_thread.start()

    app = QApplication(sys.argv)
    window = PlotWindow(result_queue)
    window.show()
    app.exec_()

    optimizer_thread.join()


if __name__ == "__main__":
    main()
