from tests.Giedre.Generate_Noise_GI import gen_noise
import subprocess


def run_test(pop, gen, noise=0, MAC=0):
    args = ['python', 'Single_Run_laje.py', str(pop), str(gen), str(noise), str(MAC)]
    subprocess.run(args)

def run_viga(pop, gen, noise=0, MAC=0):
    args = ['python', 'Single_Run_viga.py', str(pop), str(gen), str(noise), str(MAC)]
    subprocess.run(args)

#Generate noise files
# gen_noise(0.05, "laje") #Generate noise 05 file
# gen_noise(0.15, "laje") #Generate noise 15 file
# gen_noise(0.05, "viga") #Generate noise 05 file
# gen_noise(0.15, "viga") #Generate noise 15 file

# #VIGA FREQ N0
# run_viga(50,30,0)
# # VIGA FREQ N5
# run_viga(50,30,0.05)
# # VIGA FREQ N15
# run_viga(50,30,0.15)
#VIGA FREQ MAC N0
run_viga(50,30,0, 1)
# #VIGA FREQ MAC N5
# run_viga(50,30,0.05,1)
# #VIGA FREQ MAC N15
# run_viga(50,30,0.15,1)


# # PONTE FREQ NOISE = 0%
# run_test(70,30,0,0)
# # PONTE FREQ NOISE = 5%
# run_test(70,30,0.05,0)#run ponte freq N5
# PONTE FREQ NOISE = 15%
# run_test(70,30,0.15,0) #run ponte freq N15
# PONTE FREQ MAC NOISE = 0%
#run_test(70,30,0, 1)
#PONTE FREQ MAC NOISE = 5%
# run_test(70,30,0.05, 1)
# # PONTE FREQ MAC NOISE = 15%
# run_test(70,30,0.15, 1) #run ponte freq N5
