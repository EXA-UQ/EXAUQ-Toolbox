# A First Walkthrough

The purpose of this tutorial is to demonstrate the main workflow of working with the
`exauq` command line application.


## Prerequisites

* You have installed the EXAUQ-Toolbox into a virtual environment and activated that
  environment. This could be an environment created with Python's `venv` module, a
  Conda environment, or similar.
* You have simulator script living on a remote server that can be run within a Bash
  shell via SSH, according to the requirements given in
  [Writing a Simulator Script](./user-guides/simulator-script.md).
* This tutorial will create new directories and files, so you may wish to make a
  new, empty directory and navigate to it before following along with the commands
  below.


## Viewing help and documentation

To view help on the options for starting the `exauq` application, run `exauq` with the
`--help` (or `-h`) option:

```
$ exauq --help
usage: exauq [-h] [-d] [workspace]

Submit and view the status of simulations.

positional arguments:
  workspace   path to a directory for storing hardware settings and simulation results (defaults to
              '.exauq-ws')

optional arguments:
  -h, --help  show this help message and exit
  -d, --docs  open a browser at the EXAUQ documentation and exit
```

To view documentation in your browser (this website!) you can instead use the `--docs`
(or `-d`) option:

```
$ exauq --docs
```


## Starting a New Workspace

When you run `exauq` in a new directory, the CLI will guide you through creating a new _workspace_. Conceptually, a workspace corresponds to a specific 'project' requiring simulations. In practice, it’s a directory dedicated to storing settings and simulation data, enabling `exauq` to resume where it left off in subsequent sessions.

Here, we provide an updated walkthrough for starting a new workspace with line-by-line explanations of user input. Comments with numbering (e.g., `# (1)`) are provided for clarity but should not be included when following the steps in your terminal.

### Example Session

```
$ exauq
======================================================================
                     EXAUQ Command Line Interface                     
                            Version 0.1.0                             
======================================================================
----------------------------------------------------------------------
                       Workspace Initialisation                       
             A new workspace '.exauq-ws' will be set up.              
----------------------------------------------------------------------
Dimension of simulator input space: 3  # (1)
Simulator input dimension set to: 3

Press Enter to continue...
```
1. **Input Dimension**: Specify the number of input variables expected by your simulator. In this example, the simulator requires three input variables.
```
======================================================================
                     EXAUQ Command Line Interface                     
                            Version 0.1.0                             
======================================================================
----------------------------------------------------------------------
                           Interface Setup                            
----------------------------------------------------------------------

---------------------------------
  Select Interface Setup Method  
---------------------------------
  1: Interactive mode
  2: Load from file

Enter the number corresponding to your choice: 1  # (2)
```
2. **Interface Setup Method**: Choose how to configure the hardware interface:

   - Interactive mode walks you through a step-by-step setup.
   - Load from file uses a pre-configured JSON file containing interface details.

```
======================================================================
                     EXAUQ Command Line Interface                     
                            Version 0.1.0                             
======================================================================
----------------------------------------------------------------------
                 Interactive Interface Configuration                  
          Please provide details of your hardware interface           
----------------------------------------------------------------------
------------------------------------------------------
  Choose the type of hardware interface to configure  
------------------------------------------------------
  1: Unix Server Script Interface

Enter the number corresponding to your choice: 1  # (3)
Selected: Unix Server Script Interface
```
3. **Interface Type**: Select the type of hardware interface you want to configure. In this example, a Unix Server Script Interface is chosen.

```
--------------------------------------------
  Hardware Interface Configuration details  
--------------------------------------------
Hardware interface name: ExampleServer  # (4)
Hardware interface level: 1  # (5)
Host server address: server.example.com  # (6)
Host username: joe  # (7)
Path to simulator script on host: path/to/my-simulator-script.sh  # (8)
Program to run simulator script with: bash  # (9)
Use SSH agent? (Default 'no'): n  # (10)
Password for joe@server.example.com:  # (11)
Connection to server.example.com established.
Add another hardware interface? (y/n): n  # (12)
```
4. Hardware Interface Details:

-  (4) Name the hardware interface for identification.
-  (5) Assign a level to the interface.
-  (6) Provide the server address where the simulator resides.
-  (7) Enter the username for the server.
-  (8) Specify the path to the simulator script on the server.
-  (9) Indicate the program to execute the simulator script (e.g., bash for shell scripts, python for python scripts, etc.).
-  (10) Choose whether to use an SSH agent for authentication.
-  (11) If not using an SSH agent, input your server password to establish a connection.
-  (12) Decide whether to add additional hardware interfaces.

!!! note
    You can add multiple hardware interfaces to run simulations on different servers or with different programs. In this example, only one interface is added.
```
======================================================================
                     EXAUQ Command Line Interface                     
                            Version 0.1.0                             
======================================================================
----------------------------------------------------------------------
                       Workspace Setup Summary                        
----------------------------------------------------------------------
Workspace Directory: .exauq-ws
Input Dimension: 3
Interfaces Added:
Interface Name  Details                                                                                                                 
ExampleServer   user: joe; host: server.example.com; program: bash; script_path: path/to/my-simulator-scrip...; name: ExampleServ...

(exauq)>  # (13)
```
5. Workspace Summary: After completing the setup, exauq displays a summary:

- The workspace directory path.
- The simulator input dimension.
- Details of the configured hardware interfaces, including the host, user, program, and script path.

6. Command Prompt: You are now ready to use exauq to manage simulations! The command prompt `(exauq)>` indicates the CLI is ready to accept commands.

## Quitting the application

Having just started the application, let's make sure we know how to exit it! We do this
simply by entering the command `quit`, like so:

```
(exauq)> quit
$
```

## Starting the application with the same workspace

Having quit the application, let's take a look at the newly created workspace directory.
If we list the contents of the current directory (including hidden items), we see that
we have a new directory called `.exauq-ws`. This contains settings files that record the 
details provided at workspace creation:

```
$ ls -a
.  ..  .exauq-ws

$ ls .exauq-ws/
hw_params_ExampleServer.json  settings.json
```

!!! warning
    Don't go modifying the files in the workspace directory, as this will likely make the
    `exauq` application behave incorrectly. In general, we recommend leaving the
    workspace directory alone.

To start the application again with the same workspace, we simply run `exauq` within
the directory containing `.exauq-ws`. Note this will ask us for our password to connect
to the server.

```
$ exauq
Using workspace '.exauq-ws'.
Password for joe@server.example.com: 
Connection to server.example.com established.
(exauq)>
```

## The command line interpreter

Once `exauq` has been started an initialised, it is ready to receive commands from a user
to execute. Commands are entered next to the prompt `(exauq)>`, much like you enter
commands into a terminal, following the general template:

```
(exauq)> COMMAND [OPTIONS] ARGS
```

Once a command has been executed, and output is printed to the console and a new line with a prompt is presented, ready to receive
another command. The application continues to go around this 'read-evaluate-print' loop
until the application is exited.


### View in-app help

To view all available commands within the application, run the `help` command:

```
(exauq)> help

Documented commands (use 'help -v' for verbose/'help <topic>' for details):
===========================================================================
add_interface  edit     list_interfaces  resubmit      set        show  
alias          help     macro            run_pyscript  shell      submit
cancel         history  quit             run_script    shortcuts  write     

(exauq)>
```

The main commands to be aware of are `help`, `submit`, `show` and `quit`. To view help
for a particular command, run `help COMMAND`, for example:

```
(exauq)> help submit
Usage: submit [-h] [-f FILE] [-l LEVEL] [inputs [...]]

Submit jobs to the simulator.

positional arguments:
  inputs             The inputs to submit to the simulator.

optional arguments:
  -h, --help         show this help message and exit
  -f, --file FILE    A path to a csv file containing inputs to submit to the simulator.
  -l, --level LEVEL  The level of the hardware interface to use for the simulation.

(exauq)>
```

### Submitting Jobs

A _job_ consists of sending a single _input_ to the simulator. (Note that an _input_ will generally 
consist of multiple coordinates.) In our example, we specified that the simulator takes in 3-dimensional 
inputs when creating the workspace. We can submit an input to the simulator with the `submit` command, 
providing a comma-separated list of numbers. For example, to submit the input `(1.111, 2.222, 9.999)`, 
we enter the following into the application:

```
(exauq)> submit 1.111,2.222,9.999
JOBID              INPUT             
20240419183827910  (1.11, 2.22, 10.0)

(exauq)>
```

1. **Unique Job ID**: Notice that an ID has been returned for the submitted job. This is a uniquely 
generated integer used by exauq to keep track of the job status. (The ID you receive will differ.)

2. **Rounded Input Display**: The input for the job is also displayed, with coordinates rounded to 
avoid overly long output lines.


!!! note
    Although the coordinates are rounded in the INPUT table heading, the full-precision values are submitted to the simulator.

#### Handling Negative Inputs

If you try submitting an input starting with a negative number, you’ll encounter an error due to argument parsing:

```
(exauq)> submit -1,2,3
Usage: submit [-h] [-f FILE] [-l LEVEL] [inputs [...]]
Error: unrecognized arguments: -1,2,3

(exauq)>
```
To resolve this, place the input list after a --, like so:

```
(exauq)> submit -- -1,2,3
```

This will submit the input (-1.0, 2.0, 3.0) as expected.

#### Using the --level Option

The --level (or -l) option allows you to specify which hardware interface level to use for the submission. For example, to submit the input (4.444, 5.555, 6.666) using hardware level 2:

```
(exauq)> submit -l 2 4.444,5.555,6.666
JOBID              INPUT             
20240420183710123  (4.44, 5.56, 6.67)

(exauq)>
```

The default level is 1 if the --level argument is not specified.

#### Submitting Inputs from a CSV File

Instead of submitting inputs manually, you can use the --file (or -f) option to submit multiple inputs from a CSV file. For example, if you have a file named design.csv in your working directory:

```
(exauq)> submit --file design.csv
JOBID              INPUT               
20240419185014637  (1.32, -0.986, 31.4)
20240419185015374  (5.97, -3.09, -3.04)
20240419185015969  (6.98, -8.03, 0.099)
20240419185016560  (-12.0, 9.07, -2.1) 

(exauq)>
```

Each row in the file corresponds to a simulator input, and the system generates a unique job ID for each submission.


### Showing the status of jobs

Having submitted jobs, we can get the status of them by using the `show` command.

```
(exauq)> show
JOBID              INPUT                 STATUS     RESULT             
20240419183827910  (1.11, 2.22, 10.0)    Completed  13.332             
20240419185014637  (1.32, -0.986, 31.4)  Running          
20240419185015374  (5.97, -3.09, -3.04)  Running          
20240419185015969  (6.98, -8.03, 0.099)  Running          
20240419185016560  (-12.0, 9.07, -2.1)   Running          

(exauq)>
```

Notice how jobs that have successfully completed contain the value of the simulator output
in the `RESULT` column, while jobs that are still running are indicated as such.

The `exauq` application will periodically check the server for any update of job statuses.
Furthermore, it will pick up where it left off when restarting a new session with the same
workspace. For example, suppose we quit the `exauq` application and waited a while for the
jobs to finish. By starting `exauq` with the same workspace, we can check on the status of
the jobs:

```
(exauq)> quit

# wait a while...

$ exauq
Using workspace '.exauq-ws'.
Password for joe@server.example.com: 
Connection to server.example.com established.
(exauq)> show
JOBID              INPUT                 STATUS     RESULT             
20240419183827910  (1.11, 2.22, 10.0)    Completed  13.332             
20240419185014637  (1.32, -0.986, 31.4)  Completed  31.74438342        
20240419185015374  (5.97, -3.09, -3.04)  Completed  -0.1624690000000002
20240419185015969  (6.98, -8.03, 0.099)  Completed  -0.9515869999999993
20240419185016560  (-12.0, 9.07, -2.1)   Completed  -5.06375744        

(exauq)>
```

In this case, we see that the outstanding running jobs have now completed.

For a more in-depth guide on viewing information about submitted jobs, see
[Getting Information on Jobs](./user-guides/showing-jobs.md).


### Cancelling jobs

If we ever want to cancel a job before it finishes running, we can use the `cancel`
command with the job ID(s) of the job(s) we want to terminate. For example, suppose we
start off a couple of jobs:

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

Note that only jobs with a status of 'Not submitted', 'Submitted' and 'Running' can be
cancelled.

For more details on cancelling jobs, consult the
[Cancelling Jobs](./user-guides/cancelling-jobs.md) guide.


### Resubmitting jobs

You may want to resubmit a job for several reasons: perhaps a job failed, was cancelled, 
or you need to rerun a stochastic simulation with the same inputs to gather more data. 
The `resubmit` command is designed to easily handle such cases. For instance, consider 
the scenario where you have jobs that were previously cancelled and now need to be 
resubmitted.

First, let's check the status of the jobs using the `show` command to identify the job 
IDs of the cancelled jobs:

```
(exauq)> show
JOBID              INPUT                 STATUS     RESULT             
20240419183827910  (1.11, 2.22, 10.0)    Completed  13.332             
20240419185014637  (1.32, -0.986, 31.4)  Completed  31.74438342        
20240419185015374  (5.97, -3.09, -3.04)  Completed  -0.1624690000000002
20240419185015969  (6.98, -8.03, 0.099)  Completed  -0.9515869999999993
20240419185016560  (-12.0, 9.07, -2.1)   Completed  -5.06375744        
20240521102800090  (-1.0, 2.0, 3.0)      Cancelled
20240521102801154  (0.5, 0.32, -3.12)    Cancelled
```

To resubmit these cancelled jobs, use their job IDs in the `resubmit` command:

```
(exauq)> resubmit 20240521102800090 20240521102801154
OLD_JOBID          NEW_JOBID              INPUT               
20240521102800090  20240603102300450      (-1.0, 2.0, 3.0)
20240521102801154  20240603102300678      (0.5, 0.32, -3.12)

(exauq)>
```

Notice that new job IDs are generated for the resubmitted jobs. This feature allows for 
tracking the new submissions separately while maintaining a record of their origins from 
previous job IDs.

If you wish to resubmit all jobs that were either cancelled or failed, you can use filters 
with the `resubmit` command to streamline the process without needing to manually input 
each job ID. For more details on resubmitting jobs, including the use of filters, consult the
[Resubmitting Jobs](./user-guides/resubmitting-jobs.md) guide.


### Write the jobs to a CSV file

Finally, the details of the jobs can be written to a file, allowing us to use the
output of simulations in our own analysis. This is done using the `write` command:

```
(exauq)> write jobs.csv
(exauq)>
```

Note that the full precision of simulation inputs and outputs will be written to the
file. The output CSV has the same column headings that appear in the `show` command, with
the addition that the individual simulation input coordinates are put under headings
`INPUT_1`, `INPUT_2`, etc.

!!! note
    When writing to a file, the parent directory of the file needs to exist. For example,
    if we were to supply `foo/bar/jobs.csv` to `write`, then the directories `foo` and
    `bar` need to already exist.

If we quit the application and list to contents of the working directory, we see that
the `jobs.csv` file has been created:

```
(exauq)> quit
$ ls -a
.  ..  .exauq-ws  jobs.csv

```
### Viewing Interfaces

The _list_interfaces_ command displays an overview of all hardware interfaces configured in the workspace, 
including their name, level, host, user, and the number of active jobs. For example:

```
(exauq)> list_interfaces 
Name             Level  Host                 User    Active Jobs
ExampleServer    1      server.example.com   joe     1
AnotherServer    2      server2.example.com  joe     3

(exauq)>
```

This command is useful for quickly reviewing the available interfaces and monitoring job distribution.