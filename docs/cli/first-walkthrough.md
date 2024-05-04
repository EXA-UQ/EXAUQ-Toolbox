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


## Starting a new workspace

When you run `exauq` in a new directory, it will walk you through the steps for creating a
_workspace_. Conceptually, we can think of a workspace corresponding to a particular
'project' for which we need to run simulations. In practice, a workspace is nothing more
than a directory dedicated to storing settings and simulation data, which enables `exauq`
to resume where it left off next time you start it.

Below we give an example run-through of starting a new workspace, with line-by-line
explanations of steps where user input is required. (The comments with numbering, e.g. `#
(1)`, are just to help refer to the lines in the explanation and shouldn't be included
when running through for real.)

```
$ exauq
A new workspace '.exauq-ws' will be set up.
Please provide the following details to initialise the workspace...
  Dimension of simulator input space: 3  # (1)
  Host server address: server.example.com  # (2)
  Host username: joe  # (3)
  Path to simulator script on host: path/to/my-simulator-script.sh  # (4)
  Program to run simulator script with: bash  # (5)
  Use SSH agent? (Default 'no'): n  # (6)
Setting up hardware...
Password for joe@server.example.com:  # (7)
Connection to server.example.com established.
Thanks -- workspace '.exauq-ws' is now set up.
(exauq)>  # (8)
```

1. The number of input variables that the simulator expects.
2. The address to the server where the simulator lives.
3. Your username for the server.
4. The path on the server to the simulator script. _We recommend providing an absolute
   path._
5. The program to run the simulator script with. In this example, we have a shell script,
   so we use the program `bash`.
6. Whether to use an already-running SSH agent for managing authentication with private 
   keys to the server. If and agent is running and 'yes' is selected, then the `exauq`
   application will use this to connect to the server without prompting for a password.
   Otherwise, you will be prompted to enter your password to connect to the server (see
   next).
7. Enter your password to establish connection to the server. _This password is not stored
   within the workspace or anywhere else by the_ `exauq` _application._
8. After establishing a connection to the server, a directory `.exauq-ws` is created in
   the current working directory and the main `exauq` command line interpreter is entered.


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
hardware_params  settings.json
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
alias  help     macro  run_pyscript  set    shortcuts  submit
edit   history  quit   run_script    shell  show     

(exauq)>
```

The main commands to be aware of are `help`, `submit`, `show` and `quit`. To view help
for a particular command, run `help COMMAND`, for example:

```
(exauq)> help submit
Usage: submit [-h] [-f FILE] [inputs [...]]

Submit a job to the simulator.

positional arguments:
  inputs           The inputs to submit to the simulator.

optional arguments:
  -h, --help       show this help message and exit
  -f, --file FILE  A path to a csv file containing inputs to submit to the simulator.

(exauq)>
```

### Submitting jobs

A _job_ consists of sending a single _input_ to the simulator. (Note that an _input_ will
in general consist of multiple coordinates.) In our example, we specified that the
simulator takes in 3-dimensional inputs when creating the workspace. We can submit an
input to the simulator with the `submit` command, as a comma-separated list of numbers. For example, to
submit the input `(1.111, 2.222, 9.999)`, we enter the following into the application:

```
(exauq)> submit 1.111,2.222,9.999
JOBID              INPUT             
20240419183827910  (1.11, 2.22, 10.0)

(exauq)>
```

Notice that an ID has been returned for the submitted job. This is a uniquely-generated
integer for the job and is used by the `exauq` application to keep track of the status of
the job. (The ID you receive will likely differ.) The input for the job is also printed,
with coordinates rounded to avoid long lines of output.

!!! note
    Although the coordinates are rounded in the `INPUT` table heading, the full precision
    numbers are submitted to the simulator.

!!! info
    If you try submitting an input that starts with a negative number with the approach
    given above, you will encounter an error:

    ```
    (exauq)> submit -1,2,3
    Usage: submit [-h] [-f FILE] [inputs [...]]
    Error: unrecognized arguments: -1,2,3

    (exauq)>
    ```
    
    This is because the `-1` is being interpreted as an optional argument to `submit`,
    which is not what we want. To get around this, place the list of inputs after `--`,
    like so:

    ```
    (exauq)> submit -- -1,2,3
    ``` 

    This will now submit the input `(-1.0, 2.0, 3.0)` as desired.


Instead of submitting inputs manually at the prompt, there is the option to submit a
collection of inputs as read from a csv file, using the `--file` (or `-f`) option. For
example, supposing we had a csv file `design.csv` in our working directory, we could run
the command

```
(exauq)> submit --file design.csv
JOBID              INPUT               
20240419185014637  (1.32, -0.986, 31.4)
20240419185015374  (5.97, -3.09, -3.04)
20240419185015969  (6.98, -8.03, 0.099)
20240419185016560  (-12.0, 9.07, -2.1) 

(exauq)>
```

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
