import os

algorithm_list=open('algorithm.csv','r+')
mystr = algorithm_list.readline()
while mystr:
    f=open("test","w+")
    li=["#!/bin/bash\n#PBS -N neat20172018nT20\n#PBS -o /home/cs-liuy/mytestfile/neat/result/20172018\n\
#PBS -e /home/cs-liuy/mytestfile/neat/result/error\n#PBS -l nodes=1:ppn=12\n\
#PBS -l walltime=72:00:00\n#PBS -q cal-l\n#PBS -V\n#PBS -S /bin/bash\ndate\ncd /home/cs-liuy/mytestfile/neat\n"]
    f.writelines(li)   
    parameters=mystr.replace(',',' ')
    run_file='python3 /home/cs-liuy/Neat/newevolve.py %s'%(parameters)+"\n"
    f.writelines(run_file)
    f.writelines('date\n')
    f.close()
    os.system('qsub test')
    mystr = algorithm_list.readline()

algorithm_list.close()
