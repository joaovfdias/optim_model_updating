from tests.BO_ANS_TestRun import BO_run_laje
from multiprocessing import Process

def run(fun, idx):
    BO_run_laje(fun, idx)

if __name__ == '__main__':
    acq_fun = ['LCB', 'EI', 'PI']

    for i, fun in enumerate(acq_fun):
        p = Process(target=run, args=(fun, i+1))
        p.start()
        p.join()