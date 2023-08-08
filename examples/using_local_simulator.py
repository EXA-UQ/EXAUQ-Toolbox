import time

from exauq.core.modelling import Input
from tests.utilities.local_simulator import LocalSimulatorInterface

# Initialise our interface to the local simulator, using the default workspace
# directory.
interface = LocalSimulatorInterface()

# Note that, in this particular case, the local simulator runs simulations that
# take a 4-dim input. These are submitted as jobs.
job1 = Input(0.1, 0.1, 0.1, 0.1)
id1 = interface.submit_job(job1)
print(f"ID of first job submitted: {id1}")

job2 = Input(0.2, 0.2, 0.2, 0.2)
id2 = interface.submit_job(job2)
print(f"ID of second job submitted: {id2}")

# Note if we try to get the job outputs immediately, we won't get anything:
out1 = interface.get_job_output(id1)
out2 = interface.get_job_output(id2)
assert out1 is None
assert out2 is None

# However, after a while the jobs will complete and we can retrieve the outputs:
print("Checking for completion of jobs...")
jobs = [id1, id2]
while jobs:
    for id in jobs:
        out = interface.get_job_output(id)
        if out:
            print(f"  Output of job {id}: {out}")
            jobs.remove(id)
    time.sleep(1)

print("All jobs completed.")
