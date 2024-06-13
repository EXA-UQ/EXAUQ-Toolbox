# Resubmitting Jobs

The `resubmit` command within the `exauq` application allows users to resubmit simulation 
jobs that have failed, were cancelled, or need to be rerun under the same conditions. 
This user guide explains the `resubmit` command in detail, providing instructions on how 
to effectively use it in different scenarios.

## Context and Initial Job Status

After running several jobs, use the `show` command to check their status.

```
(exauq)> show -n 7
JOBID              INPUT               STATUS     RESULT 
20240424163258474  (5.0, 5.0, 5.0)     Completed  15.0  
20240424175044903  (8.0, 4.0, 5.0)     Completed  17.0          
20240424175049647  (8.0, 4.0, 8.0)     Completed  20.0
20240424185432857  (-1.0, 2.0, 3.0)    Failed
20240424185529714  (0.5, 0.32, -3.12)  Failed
```

In this example, we see that two jobs have failed. Next, we will introduce and use the `resubmit` command to handle these failed jobs.

## Basic Resubmission

To resubmit jobs, simply provide the job IDs as arguments to the `resubmit` command. 
This can be useful for jobs that have encountered errors, were terminated without 
results, or were cancelled but need to be rerun. In this case, we provide the job IDs 
of the two failed jobs. The example below shows how these failed jobs are resubmitted, 
with the command assigning new job IDs to each. This allows you to track the resubmitted 
jobs separately from their original submissions.

```
(exauq)> resubmit 20240521102800090 20240521102801154
OLD_JOBID          NEW_JOBID              INPUT               
20240521102800090  20240425120013456      (1.1, 2.2, 3.3)
20240521102801154  20240425123045879      (2.1, -1.2, 0.3) 
```

If we now run the `show` command again, we can see the new job IDs and their status:

``` 
(exauq)> show
JOBID              INPUT               STATUS     RESULT
20240424163258474  (5.0, 5.0, 5.0)     Completed  15.0
20240424175044903  (8.0, 4.0, 5.0)     Completed  17.0
20240424175049647  (8.0, 4.0, 8.0)     Completed  20.0
20240424185432857  (-1.0, 2.0, 3.0)    Failed
20240424185529714  (0.5, 0.32, -3.12)  Failed
20240425120013456  (1.1, 2.2, 3.3)     Submitted
20240425123045879  (2.1, -1.2, 0.3)    Submitted
```

## Using Filters to Select Jobs for Resubmission

To effectively manage job resubmissions, you may need to filter jobs based on their status. 
Here's an initial set of jobs to illustrate this process:

```
(exauq)> show
JOBID              INPUT               STATUS     RESULT
20240424163258474  (5.0, 5.0, 5.0)     Completed  15.0  
20240424175044903  (8.0, 4.0, 5.0)     Completed  17.0          
20240424175049647  (8.0, 4.0, 8.0)     Completed  20.0
20240424185432857  (-1.0, 2.0, 3.0)    Failed
20240424185529714  (0.5, 0.32, -3.12)  Failed
20240425120013456  (1.1, 2.2, 3.3)     Cancelled
20240425123045879  (2.1, -1.2, 0.3)    Cancelled
20240425124567890  (3.3, 3.3, 3.3)     Running
```

### Resubmitting Based on Status

If you want to resubmit jobs based on their status, such as all 'Cancelled' or 
'Failed' jobs, you can use the `--status` option with the `resubmit` command. 
For example, to resubmit all jobs that have failed:

```
(exauq)> resubmit --status=failed
OLD_JOBID          NEW_JOBID              INPUT               
20240424185432857  20240603102300450      (-1.0, 2.0, 3.0)
20240424185529714  20240603102300678      (0.5, 0.32, -3.12)
```

To resubmit jobs with multiple statuses, such as 'Failed' and 'Cancelled', you 
can provide both statuses with the `--status` option:

```
(exauq)> resubmit --status=failed,cancelled
OLD_JOBID          NEW_JOBID              INPUT               
20240424185432857  20240603102300450      (-1.0, 2.0, 3.0)
20240424185529714  20240603102300678      (0.5, 0.32, -3.12)
20240425120013456  20240603102300901      (1.1, 2.2, 3.3)
20240425123045879  20240603102301234      (2.1, -1.2, 0.3)
```

This command resubmits all jobs that have either failed or been cancelled, 
assigning new job IDs to each for separate tracking.

### Excluding Certain Statuses

Alternatively, if you want to resubmit all jobs except those in certain statuses, use 
the `--status-not` option. For example, to resubmit all jobs that are not 'Completed':

```
(exauq)> resubmit --status-not=completed
OLD_JOBID          NEW_JOBID              INPUT               
20240424185432857  20240603103501234      (-1.0, 2.0, 3.0)
20240424185529714  20240603103501789      (0.5, 0.32, -3.12)
20240425120013456  20240603103502156      (1.1, 2.2, 3.3)
20240425123045879  20240603103502567      (2.1, -1.2, 0.3)
20240425124567890  20240603103502978      (3.3, 3.3, 3.3)
```

This command resubmits all jobs that are not completed, assigning new job IDs to each for 
separate tracking.

!!! note
    The `--status` and `--status-not` options can be used together to filter jobs based on 
    multiple statuses. For example, to resubmit all jobs that are 'Failed' or 'Cancelled' 
    but not 'Completed', use the following command:

    ```
    (exauq)> resubmit --status=failed,cancelled --status-not=completed
    ```
!!! warning
    Excluding a status using `--status-not` will include all other statuses, including
    non-terminal statuses like 'Running' and 'Submitted'. Be sure to consider the full
    range of statuses when using this option.

### Resubmitting Jobs That Terminated Without Result

For jobs that have ended (i.e., are no longer running) but have not produced a result, 
the `--twr` (terminated without result) option is especially valuable. This option 
specifically targets jobs with terminal statuses that did not generate results, such 
as 'Failed', 'Cancelled', and 'Failed Submit'.

Here is the initial set of jobs, including some that terminated without result:

```
(exauq)> show
JOBID              INPUT               STATUS          RESULT
20240424163258474  (5.0, 5.0, 5.0)     Completed       15.0  
20240424175044903  (8.0, 4.0, 5.0)     Completed       17.0          
20240424175049647  (8.0, 4.0, 8.0)     Completed       20.0
20240424185432857  (-1.0, 2.0, 3.0)    Failed
20240424185529714  (0.5, 0.32, -3.12)  Failed
20240425120013456  (1.1, 2.2, 3.3)     Cancelled
20240425123045879  (2.1, -1.2, 0.3)    Cancelled
20240425124567890  (3.3, 3.3, 3.3)     Running
20240426120013457  (4.4, 4.4, 4.4)     Failed Submit
20240426123045880  (5.5, 5.5, 5.5)     Failed Submit
```

Using the `--twr` option, you can resubmit all jobs that terminated without producing 
a result:

```
(exauq)> resubmit --twr
OLD_JOBID          NEW_JOBID              INPUT               
20240424185432857  20240603103500456      (-1.0, 2.0, 3.0)
20240424185529714  20240603103501984      (0.5, 0.32, -3.12)
20240425120013456  20240603103502457      (1.1, 2.2, 3.3)
20240425123045879  20240603103502912      (2.1, -1.2, 0.3)
20240426120013457  20240603103503345      (4.4, 4.4, 4.4)
20240426123045880  20240603103503789      (5.5, 5.5, 5.5)
```

This command resubmits all jobs that ended without producing an output, streamlining 
the process of handling unsuccessful job runs by specifically targeting those jobs that 
did not generate results, unlike the `--status` option which targets jobs based on 
their end state.

!!!note
    The `--twr` option overrides any filters applied to statuses such as `--status` and 
    `--status-not`. For example, the `--status-not=failed` filter is overridden by the 
    `--twr` option, resulting in the inclusion of jobs that have failed but did not produce
    a result.
