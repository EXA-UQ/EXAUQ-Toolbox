# Writing a Simulator Script

By default, the `exauq` application is set up to connect with a server via SSH and run
a simulator script within a Bash shell. In order to run simulations, the simulator script
needs to assume a particular command line interface, which is described in what follows.

In order to run a simulation, the script should take in exactly two command line
arguments:

* The first argument should be the path to a CSV file containing the simulation input
  data. The script should read the data from this file.
* The second argument should be the path to a text file that the script will write the
  simulation output to.

The simulator script is then run within Bash according to the general form:

``` console
$ PROGRAM path/to/simulator-script path/to/simulation-input.csv path/to/simulation-output.txt
```
