import os
import csv

script_fmt = ( "NUMQUBITS=%d\n"
               "GATESIZE=%d\n"
               "DEPTH=%d\n"
               "POINTS=%d\n"
               "TIMEOUT=%d\n"
               "python rand_seq.py $NUMQUBITS $GATESIZE $DEPTH $POINTS $TIMEOUT\n" )

script_name_fmt = "run_%dq_%dg_%dd_%dp.sh"

with open( "exp.csv", newline='' ) as csvfile:
        r = csv.reader( csvfile )
        for i, row in enumerate( r ):
            # Skip Header
            if i == 0:
                continue

            # Write script to file
            script = script_fmt % tuple( [ int( x ) for x in row ] )
            script_name = script_name_fmt % tuple( [ int( x ) for x in row[:-1] ] )
            
            with open( script_name, "w" ) as f:
                f.write( script )

            # Make script executable
            mode = os.stat( script_name ).st_mode
            mode |= (mode & 0o444) >> 2
            os.chmod( script_name, mode )

