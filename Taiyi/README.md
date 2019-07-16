
# How to use TaiYi to run Online Neat
-------------

> Email: [liur3@mail.sustech.edu.cn](liur3@mail.sustech.edu.cn) 
 
<!---->
## *about files

If you only need to run one task or submit a task to the server, you need to creat the folder of output and error, and then you just need to change file "test" based on the instruction in that file, and run this:
```sh
bsub <test
```

If your want to check the stats of the tasks, use:
```sh
bjobs +JOBID
```

Four states of Jobs in Taiyi:
- PEND - the job is in the queue, waiting to be scheduled
- PSUSP - the job was submitted, but was put in the suspended state (ineligible to run)
- RUN - the job has been granted an allocation. If it’s a batch job, the batch script has been run
- DONE - the job has completed successfully