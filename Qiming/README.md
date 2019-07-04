
# How to use QiMing to run Online Neat
-------------

> Email: [liur3@mail.sustech.edu.cn](liur3@mail.sustech.edu.cn) 
 
<!---->
## *about files

If you only need to run one task or submit a task to the server, you just need to change file "test", and run this:
```sh
qsub test
```

if you want to submit many tasks at the same time, you need to change "run.py", and add your parameters in "algorithm.csv" and run these :
```sh
mpicc -o mytest multi.c
python3 run.py   #let him produce files "test" 
```

If your want to check the stats of the tasks, use:
```sh
qstat
```

If your want to delete the tasks, use:
```sh
qdel + tasknumber
```
