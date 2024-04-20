# Writing a Simulator Script

By default, the `exauq` application is set up to connect with a server via SSH and run
a simulator script within a Bash shell. In order to run simulations, the simulator script
needs to assume a particular command line interface. Specifically, the script should take
a JSON file as its sole command line argument. It should expect this JSON file to consist
of a single object which gives the path to a CSV file containing the simulation input data
(under the name `"input_file"`) and the path to a text
file to write the simulation output to (under the name `"output_file"`). A template for the
content of this JSON file is given below.

``` json title="JSON file template for a simulation"
{
    "input_file": "path/to/simulation-input.csv",
    "output_file": "path/to/simulation-output.txt"
}
```

The simulator script should read this JSON file, use it to read the input data for
the simulation from the given CSV file and then write the result to the given output file.
The simulator script is then run within Bash according to the general form:

``` console
$ PROGRAM path/to/simulator-script path/to/simulation-file.json
```
