"""
exauq.sim_management
=====================

-------------------------------------------------------------------------------------------
The `exauq.sim_management` package provides the infrastructure for managing simulation
jobs, including job submission, monitoring, cancellation, and result retrieval across
local and remote hardware environments. It abstracts the complexities of interacting
with various computational resources, enabling seamless execution of simulations in
distributed systems.

The package supports job orchestration through flexible hardware interfaces, including
SSH-based connections, and maintains a detailed log of all submitted jobs, their
statuses, and outputs. This facilitates efficient simulation workflows, especially
for uncertainty quantification tasks at scale.

-------------------------------------------------------------------------------------------
Modules
=======

[`hardware`][exauq.sim_management.hardware]:
    Defines abstract and concrete classes for hardware interfaces, allowing job
    submission, status tracking, and output retrieval on both local and remote
    systems. Includes SSH-based interfaces for secure remote job management.

[`jobs`][exauq.sim_management.jobs]:
    Contains classes representing simulation jobs (`Job`) and job identifiers (`JobId`).
    Handles job creation, data validation, and encapsulates job-related metadata.

[`simulators`][exauq.sim_management.simulators]:
    Provides tools for managing simulation workflows, including logging simulation
    inputs and outputs, monitoring job progress, and handling job status updates.
    Supports integration with various simulation backends and hardware interfaces.

[`types`][exauq.sim_management.types]:
    Defines reusable type aliases, such as `FilePath`, to represent common data
    structures like file paths, enhancing code readability and type safety.

-------------------------------------------------------------------------------------------
"""
