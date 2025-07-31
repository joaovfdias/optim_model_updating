import sys
import csv
import random
import time
import os
import threading
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

OUTPUT_FILE = "fitness_log.csv"
NUM_ITERATIONS = 50
SLEEP_TIME = 0.5



def run_optimization_simulation():
    with open(OUTPUT_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["iteration", "fitness"])

    for i in range(NUM_ITERATIONS):
        fitness = random.uniform(0.1, 1.0) / (i + 1)
        with open(OUTPUT_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, fitness])
        print(f"[OPTIMIZER] Iteration {i}: fitness = {fitness:.5f}")
        time.sleep(SLEEP_TIME)



class PlotWindow(QMainWindow):
    def _init_(self):
        super().__init__()
        self.setWindowTitle("Live Fitness Plot")
        self.canvas = FigureCanvas(Figure(figsize=(6, 4)))
        self.ax = self.canvas.figure.subplots()
        self.ax.set_title("Fitness Evolution")
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Fitness")
        self.line, = self.ax.plot([], [], '-o', label="Fitness")
        self.ax.legend()

        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.x_data = []
        self.y_data = []
        self.last_iteration = -1

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(300)  # Update every 300ms

    def update_plot(self):
        try:
            if not os.path.exists(OUTPUT_FILE):
                return

            df = pd.read_csv(OUTPUT_FILE)
            new_data = df[df["iteration"] > self.last_iteration]
            if not new_data.empty:
                self.x_data.extend(new_data["iteration"])
                self.y_data.extend(new_data["fitness"])
                self.last_iteration = max(new_data["iteration"])

                self.line.set_data(self.x_data, self.y_data)
                self.ax.relim()
                self.ax.autoscale_view()
                self.canvas.draw()

            if self.last_iteration >= NUM_ITERATIONS - 1:
                self.timer.stop()

        except Exception as e:
            print(f"[PLOTTER] Error: {e}")



def main():
    # Start optimizer in background thread
    optimizer_thread = threading.Thread(target=run_optimization_simulation)
    optimizer_thread.start()

    # Start GUI in main thread
    app = QApplication(sys.argv)
    window = PlotWindow()
    window.show()
    app.exec_()

    optimizer_thread.join()


# if _name_ == "_main_":
#     main()

main()