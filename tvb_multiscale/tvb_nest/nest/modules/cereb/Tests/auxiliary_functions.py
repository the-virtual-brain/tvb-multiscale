import time

def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        tempo = float(time.time()) - float(startTime_for_tictoc)
        print("Elapsed time is %.3f seconds" % tempo)
    else:
        print("Toc: start time not set")
