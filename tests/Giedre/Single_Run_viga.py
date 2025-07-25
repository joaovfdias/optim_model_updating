import sys
from tests.Giedre.viga.GA_ANS_VIGA_GI import run_viga

#viga, frequencias
pop = int(sys.argv[1])
gen = int(sys.argv[2])
noise = float(sys.argv[3]) if len(sys.argv) > 3 else 0
MAC = int(sys.argv[4]) if len(sys.argv) > 4 else None

run_viga(pop=pop, gen=gen, noise=noise, MAC= MAC)