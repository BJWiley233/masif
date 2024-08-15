import os
import sys
from subprocess import Popen, PIPE

from input_output.read_msms import read_msms
from triangulation.xyzrn import output_pdb_as_xyzrn, output_pdb_as_xyzrn2
from default_config.global_vars import msms_bin 
from default_config.masif_opts import masif_opts
import random

# Pablo Gainza LPDI EPFL 2017-2019
# Calls MSMS and returns the vertices.
# Special atoms are atoms with a reduced radius.
def computeMSMS(pdb_file,  protonate=True):
    randnum = random.randint(1,10000000)
    file_base = masif_opts['tmp_dir']+"/msms_"+str(randnum)
    out_xyzrn = file_base+".xyzrn"

    if protonate:        
        output_pdb_as_xyzrn(pdb_file, out_xyzrn)
    else:
        print("Error - pdb2xyzrn is deprecated.")
        sys.exit(1)
    # Now run MSMS on xyzrn file
    FNULL = open(os.devnull, 'w')
    args = [msms_bin, "-density", "3.0", "-hdensity", "3.0", "-probe",\
                    "1.5", "-if",out_xyzrn,"-of",file_base, "-af", file_base]
    print(msms_bin+" "+args)
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()

    vertices, faces, normals, names = read_msms(file_base)
    areas = {}
    ses_file = open(file_base+".area")
    next(ses_file) # ignore header line
    for line in ses_file:
        fields = line.split()
        areas[fields[3]] = fields[1]

    print('file_base', file_base)
    # Remove temporary files. 
    # os.remove(file_base+'.area')
    # os.remove(file_base+'.xyzrn')
    # os.remove(file_base+'.vert')
    # os.remove(file_base+'.face')
    return vertices, faces, normals, names, areas

def computeMSMS2(pdb_file,  protonate=True, probe='1.5', density='3.0', hdens='3.0',pout=False):
    randnum = random.randint(1,10000000)
    file_base = masif_opts['tmp_dir']+"/msms_"+str(randnum)
    out_xyzrn = file_base+".xyzrn"
    # print('output_pdb_as_xyzrn2', out_xyzrn)
    if protonate:        
        print('protonating output_pdb_as_xyzrn2', pdb_file, out_xyzrn)
        output_pdb_as_xyzrn2(pdb_file, out_xyzrn, pout)
    else:
        print("Error - pdb2xyzrn is deprecated.")
        sys.exit(1)
    # Now run MSMS on xyzrn file
    FNULL = open(os.devnull, 'w')
    args = [msms_bin, "-density", density, "-hdensity", hdens, "-probe",\
                    probe, "-if",out_xyzrn,"-of",file_base, "-af", file_base]
    #print msms_bin+" "+`args`
    print(args)
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()

    vertices, faces, normals, names = read_msms(file_base)
    areas = {}
    ses_file = open(file_base+".area")
    next(ses_file) # ignore header line
    for line in ses_file:
        fields = line.split()
        areas[fields[3]] = fields[1]

    print('file_base', file_base)
    # Remove temporary files. 
    # commented = True
    commented = False
    if commented:
        print("################### MAKE SURE TO UNCOMMENT REMOVING TEMP FILES!!!!!! ###################")
    os.remove(file_base+'.area')
    os.remove(file_base+'.xyzrn')
    os.remove(file_base+'.vert')
    os.remove(file_base+'.face')
    return vertices, faces, normals, names, areas