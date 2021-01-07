import os
import csv

script_fmt = ( "NUMQUBITS=%d\n"
               "GATESIZE=%d\n"
               "DEPTH=%d\n"
               "TIMEOUT=%d\n"
               "TESTQFACTOR=%s\n"
               "python fixed_time_exp.py $NUMQUBITS $GATESIZE $DEPTH $POINTS $TIMEOUT\n" )

script_name_fmt = "run_%s_%dq_%dg_%dd_%ds.sh"

timeout = 7200
for numqubits in [4, 5, 6, 7, 8]:
    for gatesize in [1, 2, 3]:
        for depth in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            
            # Write qfactor script to file
            script = script_fmt % ( numqubits, gatesize, depth, timeout, "--testqfactor" )
            script_name = script_name_fmt % ( "qfactor", numqubits, gatesize, depth, timeout )
            
            with open( script_name, "w" ) as f:
                f.write( script )

            # Make script executable
            mode = os.stat( script_name ).st_mode
            mode |= (mode & 0o444) >> 2
            os.chmod( script_name, mode )

            # Write comparison script to file
            script = script_fmt % ( numqubits, gatesize, depth, timeout, "\"\"" )
            programname = "qfast" if gatesize > 1 else "qsearch"
            script_name = script_name_fmt % ( programname, numqubits, gatesize, depth, timeout )
            
            with open( script_name, "w" ) as f:
                f.write( script )

            # Make script executable
            mode = os.stat( script_name ).st_mode
            mode |= (mode & 0o444) >> 2
            os.chmod( script_name, mode )
