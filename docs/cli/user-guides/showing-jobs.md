# Getting Information on Jobs

The `show` command within the `exauq` application is used to get information about
simulation jobs that have been submitted as part of the current workspace. Users can
either specify jobs by ID to only show information for these jobs, or instead apply
filters to all jobs within the workspace. This page discusses the `show` command in more
detail.

## Show only specific jobs

To show information about specific jobs, simply provide the IDs for the jobs as arguments:

```
(exauq)> show 20240420133041807 20240420184546233
JOBID              INPUT            STATUS     RESULT
20240420133041807  (1.0, 2.0, 3.0)  Completed  6.0   
20240420184546233  (0.0, 0.0, 0.0)  Failed           
```

If no job IDs are supplied, then the 50 most recent jobs will be displayed:

```
(exauq)> show
JOBID              INPUT            STATUS     RESULT
20240420133041807  (1.0, 2.0, 3.0)  Completed  6.0   
20240420184546233  (0.0, 0.0, 0.0)  Failed
... # Lots more jobs here ...

```

## Limiting the number of jobs shown

To view the status of all jobs stored in the workspace, use the `--all` (or `-a`) option:

```
(exauq)> show --all
JOBID              INPUT            STATUS     RESULT
20240420133041807  (1.0, 2.0, 3.0)  Completed  6.0   
20240420184546233  (0.0, 0.0, 0.0)  Failed
... # Show ALL the jobs ...

```

To view only a certain number of jobs (counting backwards), use the `--n-jobs` (or `-n`)
option. For example, to view only the last 5 jobs:

```
(exauq)> show -n 5
JOBID              INPUT            STATUS     RESULT
20240424101653771  (1.0, 1.0, 1.0)  Completed  3.0   
20240424101654578  (2.0, 2.0, 2.0)  Completed  6.0   
20240424163258474  (5.0, 5.0, 5.0)  Completed  15.0  
20240424175044903  (8.0, 4.0, 5.0)  Running          
20240424175049647  (8.0, 4.0, 8.0)  Submitted          

```

## Filtering on status

To show all jobs having one of a collection of statuses, use the optional argument
`--status` (or `-s`) followed by a comma-separated list of the statuses to be shown. To
show all jobs _not_ having one of a collection of statuses, use the optional argument
`--status-not` (or `-S`) instead.

For example, to show all jobs with a status of 'Submitted' or 'Running':

```
(exauq)> show --status=submitted,running
JOBID              INPUT            STATUS     RESULT
20240424175044903  (8.0, 4.0, 5.0)  Running          
20240424175049647  (8.0, 4.0, 8.0)  Submitted        
```

To show all jobs except those that are 'Completed' or 'Submitted':

```
(exauq)> show --status-not=completed,failed
JOBID              INPUT            STATUS     RESULT
20240420184546233  (0.0, 0.0, 0.0)  Failed           
20240420184607856  (1.0, 1.0, 1.0)  Cancelled        
20240424175044903  (8.0, 4.0, 5.0)  Running          
```

## Filtering on simulation output

It's also possible to filter jobs on whether there is a simulation output ('result')
recorded, using the `--result` (or `-r`) option.

* To show only jobs that have an output, use `--result` on its own or `--result=true`:
  
  ```
  (exauq)> show --result
  JOBID              INPUT            STATUS     RESULT
  20240420133041807  (1.0, 2.0, 3.0)  Completed  6.0   
  20240424095404029  (9.0, 9.0, 9.0)  Completed  27.0  
  20240424101653771  (1.0, 1.0, 1.0)  Completed  3.0   
  20240424101654578  (2.0, 2.0, 2.0)  Completed  6.0   
  20240424163258474  (5.0, 5.0, 5.0)  Completed  15.0  
  ```

* To show only jobs that _don't_ have an output, use `--result=false`:
  
  ```
  (exauq)> show --result=false
  JOBID              INPUT            STATUS     RESULT
  20240420184546233  (0.0, 0.0, 0.0)  Failed           
  20240420184607856  (1.0, 1.0, 1.0)  Cancelled        
  20240424175044903  (8.0, 4.0, 5.0)  Running          
  20240424175049647  (8.0, 4.0, 8.0)  Submitted      
  ```

* To show jobs regardless of whether there is a simulation output, simply don't apply the
  `--result` option (or set it to the empty string `''`).

## Combining filters

The filtering options described can be combined to make more refined filters. The general
rule is that multiple filters are combined with the 'and' operation, to successively apply
more restrictive filtering.

For example, to view all jobs that don't have a simulation output but haven't failed:

```
(exauq)> show --result=false --status-not=failed
JOBID              INPUT            STATUS     RESULT
20240420184607856  (1.0, 1.0, 1.0)  Cancelled        
20240424175044903  (8.0, 4.0, 5.0)  Running          
20240424175049647  (8.0, 4.0, 8.0)  Submitted      
```

Or to view only the last 3 completed jobs:

```
(exauq)> show --status=completed -n 3
JOBID              INPUT            STATUS     RESULT
20240424101653771  (1.0, 1.0, 1.0)  Completed  3.0   
20240424101654578  (2.0, 2.0, 2.0)  Completed  6.0   
20240424163258474  (5.0, 5.0, 5.0)  Completed  15.0  
```

## A shortcut: jobs terminated without result

It is common to want to view all jobs that have terminated (i.e. are no longer running)
but for which there is no output. These can be displayed with the 'terminated without
result' option, `--twr` (or `-x`):

```
(exauq)> show --twr
JOBID              INPUT            STATUS     RESULT
20240420184546233  (0.0, 0.0, 0.0)  Failed           
20240420184607856  (1.0, 1.0, 1.0)  Cancelled        
```

Depending on your use case, this can be useful for identifying jobs that may need
resubmitting.

!!! note

    The `--twr` option overrides any filters applied to statuses or simulation outputs via
    `--status`, `--status-not` and `--result` arguments. Notice how the job with status
    'Failed' is _not_ removed in the following, because `--status-not=failed` is overridden
    by `--twr`:

    ```
    (exauq)> show --twr --status-not=failed
    JOBID              INPUT            STATUS     RESULT
    20240420184546233  (0.0, 0.0, 0.0)  Failed           
    20240420184607856  (1.0, 1.0, 1.0)  Cancelled        
    ``` 