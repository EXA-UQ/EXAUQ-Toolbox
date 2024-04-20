# Filtering Jobs

TODO: complete example output

By default, only the 50 most recent jobs are displayed by `show`. To view the status of
all jobs stored in the workspace, use the `--all` (or `-a`) option:

``` console
(exauq)> show --all
```

To view only a certain number of jobs (counting backwards), use the `--n` (or just `-n`)
option. For example, to view only the last 5 jobs:

``` console
(exauq)> show -n 5
```

To only show the status of a particular job, supply the job ID as an argument to `show`:

``` console
(exauq)> show 20240405005500567
```

It is also possible to filter on various columns of the output.

To show all jobs having one of a collection of statuses, use the optional argument
`--status` (or `-s`) followed by a comma-separated list of the statuses to be shown. To
show all jobs _not_ having one of a collection of statuses, use the optional argument
`--status-not` (or `-S`) instead.

For example, to show all jobs with a status of 'Submitted', 'Running' or 'Completed':

``` console
(exauq)> show --status=submitted,running,completed
```

To show all jobs except those that are 'Completed' or 'Failed':

``` console
(exauq)> show --status-not=completed,failed
```

To only show jobs that have a result, use the `--result` (or `-r`) option. To only show
jobs that _don't_ have a result, use the `--result-none` (or `-R`) option.

``` console
(exauq)> show --result-none
```

Note this can be combined with other filters, where combination is the 'AND' operator.
For example, to view all jobs that don't have a result but haven't failed, we could run:

``` console
(exauq)> show -R -S failed
```

It is common to want to view all jobs that have terminated (i.e. are no longer running)
but for which there is no output. These can be displayed with the 'terminated without
result' option, `--twr` (or `-x`):

``` console
(exauq)> show --twr
```
