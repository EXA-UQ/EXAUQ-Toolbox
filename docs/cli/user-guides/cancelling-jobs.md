# Cancelling Jobs

The `cancel` command within the `exauq` application is used to cancel simulation jobs
that have not terminated. This can be useful if a job was started accidentally, or with
the wrong simulator inputs, etc.

## Which jobs can be cancelled?

Only jobs that have a _non-terminal_ status can be cancelled, i.e. jobs that are either
running or waiting to be run. More precisely, a job may be cancelled unless it has one
of the following terminal statuses:

* Completed
* Failed
* Failed submit
* Cancelled


## Cancelling jobs

Suppose we start off a couple of jobs:

```
(exauq)> submit -- -1,2,3 0.5,0.32,-3.12
JOBID              INPUT             
20240521102800090  (-1.0, 2.0, 3.0)  
20240521102801154  (0.5, 0.32, -3.12)

(exauq)> 
```

We can see that the status of the jobs is 'Running':

```
(exauq)> show 20240521102800090 20240521102801154
JOBID              INPUT               STATUS   RESULT
20240521102800090  (-1.0, 2.0, 3.0)    Running        
20240521102801154  (0.5, 0.32, -3.12)  Running        

(exauq)>
```

To cancel the new jobs, we supply the corresponding job IDs. The returned output confirms
their cancellation:

```
(exauq)> cancel 20240521102800090 20240521102801154
JOBID              INPUT               STATUS   
20240521102800090  (-1.0, 2.0, 3.0)    Cancelled
20240521102801154  (0.5, 0.32, -3.12)  Cancelled

(exauq)> 
```

If we try cancelling a job with one of the terminal statuses, then this job will be
ignored and a message will be printed to the screen. Similarly, if we provide a job ID
that doesn't exist, this ID will be ignored and warning will be printed to standard error.
Any other jobs that are provided alongside such IDs will be cancelled as usual. 

For example, if we try cancelling one of the jobs that we just cancelled along with a
made up job ID, we get the following:

```
(exauq)> cancel 20240521102800090 00000000000000000
The following jobs have already terminated and were not cancelled:
  20240521102800090
Warning: Could not find jobs with the following IDs:
  00000000000000000
(exauq)>
```