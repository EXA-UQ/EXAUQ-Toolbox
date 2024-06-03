# Resubmitting Jobs

The `resubmit` command within the `exauq` application allows users to resubmit simulation 
jobs that have failed, were cancelled, or need to be rerun under the same conditions. 
This user guide explains the `resubmit` command in detail, providing instructions on how 
to effectively use it in different scenarios.

## Basic Resubmission

To resubmit jobs, simply provide the job IDs as arguments to the `resubmit` command. 
This can be useful for jobs that have encountered errors, were terminated without 
results, or were cancelled but need to be rerun.

```
(exauq)> resubmit 20240521102800090 20240521102801154
OLD_JOBID          NEW_JOBID              INPUT               
20240521102800090  20240603102300450      (-1.0, 2.0, 3.0)
20240521102801154  20240603102300678      (0.5, 0.32, -3.12)
```

In the example above, two cancelled jobs are resubmitted and the command assigns new 
job IDs to each, allowing you to track the resubmitted jobs separately from their 
original submissions.

## Using Filters to Select Jobs for Resubmission

### Resubmitting Based on Status

If you want to resubmit jobs based on their status, such as all 'Cancelled' or 'Failed' 
jobs, you can use the `--status` option with the `resubmit` command. For example, to 
resubmit all jobs that have failed:

```
(exauq)> resubmit --status=failed
OLD_JOBID          NEW_JOBID              INPUT               
20240419185014637  20240603102300450      (1.32, -0.986, 31.4)
20240419185015969  20240603102300894      (6.98, -8.03, 0.099)
```

### Excluding Certain Statuses

Alternatively, if you want to resubmit all jobs except those in certain statuses, use 
the `--status-not` option. For example, to resubmit all jobs that are not 'Completed':

```
(exauq)> resubmit --status-not=completed
OLD_JOBID          NEW_JOBID              INPUT               
20240419185015374  20240603103501234      (5.97, -3.09, -3.04)
20240419185015969  20240603103501789      (6.98, -8.03, 0.099)
20240521102800090  20240603103502156      (-1.0, 2.0, 3.0)
20240521102801154  20240603103502567      (0.5, 0.32, -3.12)
```

### Resubmitting Jobs That Terminated Without Result

For jobs that have ended (i.e., are no longer running) but have not produced a result, 
the --twr (terminated without result) option is especially valuable:

```
(exauq)> resubmit --twr
OLD_JOBID          NEW_JOBID              INPUT               
20240420184546233  20240603103500456      (0.0, 0.0, 0.0)
20240420184607856  20240603103501984      (1.0, 1.0, 1.0)
```

This command resubmits all jobs that ended without producing an output, streamlining 
the process of handling unsuccessful job runs.

## Summary

The `resubmit` command in `exauq` provides flexible and powerful tools for managing 
the lifecycle of simulation jobs. By understanding and utilizing the various options 
available, users can ensure that their simulations are efficiently managed and that 
data collection can continue seamlessly, even in the face of failed or incomplete jobs.