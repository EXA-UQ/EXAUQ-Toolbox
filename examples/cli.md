# exauq cli example usage

## Workspaces

We assume users may have multiple projects for which they want to use the `exauq` tool
to manage simulator jobs. Data that needs to persist between `exauq` sessions should be
stored in a `exauq`-bespoke 'workspace' folder, one for each project. This
workspace folder should live alongside other scripts, analysis etc. that the user has
written for the given project. By default, it will be called `.exauq`, though the user
can specify a different name (or even path) if they desire. It is expected that the user
won't go poking around in the workspace folder ordinarily.

The workspace folder contains:

- Details about the hardware interface (e.g. kind of interface, IP address, username,...).
  This is so the user doesn't have to re-enter these details when starting a new session,
  once these details have been initially written.

- The simulations log.

Interaction with workspace directories works as follows:

- If no workspace directory exists, then the command:
  
  ```bash
  # Create default .exauq directory in cwd
  $ exauq

  # Create bespoke directory
  $ exauq path/to/my-workspace
  ```
  
  will create a new workspace directory and walk the user through initialisation.

- If a workspace directory exists, then `exauq` will look for the default in the cwd or
  use the one provided by the user. Upon finding it, it reads in the data stored there
  and allows the user to pick up where they left off.

  ```bash
  # Will use .exauq in cwd
  $ exauq

  # Use bespoke workspace dir
  $ exauq path/to/my-workspace
  ```

## Command format

Prompt is:
```
(exauq)
```

Within the app, commands issued according to the template:

```
(exauq) COMMAND [OPTIONS] ARGS
```

## Job submission

Submit a new design point for simulator evaluation

```
(exauq) submit 1.1,2.2,3.3
JOBID   Input
99      1.12,2.23,-1.3

(exauq)
```

Or submit a batch of inputs from a csv file (optional input column, rounded to 2dp to avoid
overflow lines of terminal text)

```
(exauq) submit path/to/design.csv
JOBID   BATCHID     INPUT
99      99          1.12,2.23,-1.3
100     99          3.12,4.23,-3.3
101     99          1.12,2.23,-4.3
102     99          0.12,5.53,-1.3
103     99          1.12,2.03,-0.3

(exauq)
```

## Job status check

Check the status of a job using the job ID
```
(exauq) status 99
JOBID   STATUS
99      RUNNING

(exauq)
```

... Or for a whole batch of jobs
```
(exauq) status -b 99
JOBID   BATCHID     STATUS
99      99          RUNNING
100     99          CANCELLED
101     99          FAILED
102     99          RUNNING
103     99          COMPLETED

(exauq)
```

## Output retrieval

Retrieve the output for a single job (if available). Give full precision of output.
```
(exauq) result 99
JOBID   STATUS      OUTPUT
99      COMPLETED   3.141592654

(exauq)
```

If not finished...

```
(exauq) result 99
JOBID   STATUS      OUTPUT
99      RUNNING

(exauq)
```

Get outputs for each job in a batch (if available)

```
(exauq) result -b 99
JOBID   BATCHID     STATUS      OUTPUT
99      99          RUNNING     
100     99          CANCELLED     
101     99          FAILED     
102     99          RUNNING     
103     99          COMPLETED   3.141592654

(exauq)
```

Option to write csv output
```
(exauq) result -f path/to/output.csv -b 99

(exauq)
```

csv file (output last column):
```txt
JOBID,X1,X2,X3,Y
99,1.12454,2.23646,-1.36464,
100,3.1245454,4.236578,-3.375753,
101,1.123384,2.2343675,-4.3452326,
102,0.1236775,5.5333678,-1.300864,
103,1.1236732,2.03321115,-0.3254656,3.141592654
```

## Other convenience features for job management

Get status of **all** terminal jobs that haven't completed successfully

(Note: might be nice to extend this to filter jobs on arbitrary statuses)

```
(exauq) status -x
JOBID   BATCHID     STATUS
100     99          CANCELLED
101     99          FAILED

(exauq)
```

Resubmit a job that didn't complete successfully
```
(exauq) submit -r 100
JOBID   OLD_JOBID   Input
105     (100)       3.12,4.23,-3.3

(exauq)
```

Resubmit a batch of jobs that all failed (might not need this)
```
(exauq) submit -rb 99
Resubmitting job batch 99 with new IDs:
JOBID   BATCHID OLD_JOBID   OLD_BATCHID Input
104     104     (99)        (99)        1.12,2.23,-1.3
105     104     (100)       (99)        3.12,4.23,-3.3
106     104     (101)       (99)        1.12,2.23,-4.3
107     104     (102)       (99)        0.12,5.53,-1.3
108     104     (103)       (99)        1.12,2.03,-0.3

(exauq)
```

Resubmit **all** jobs that have failed or been cancelled
```
(exauq) submit -s
JOBID   OLD_JOBID   OLD_STATUS      Input
106     (100)       (CANCELLED)     3.12,4.23,-3.3
107     (101)       (FAILED)        1.12,2.23,-4.3

(exauq)
```