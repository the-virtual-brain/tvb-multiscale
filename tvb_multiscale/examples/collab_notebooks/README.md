# Running jobs on CSCS infrastructure from collab

### 1. Prepare UNICORE client api
PYUNICORE client library is available on PYPI. In order to use it you have to install it using:
```sh
$ pip install pyunicore
```
Next step is to configure client registry and what supercomputer to use
```sh
tr = unicore_client.Transport(oauth.get_token())
r = unicore_client.Registry(tr, unicore_client._HBP_REGISTRY_URL)
# use "DAINT-CSCS" -- change if another supercomputer is prepared for usage
client = r.site('DAINT-CSCS')
```

### 2. Prepare job submission
In this step we have to prepare a JSON object which will be used in the job submission process.
```sh
# What job will execute (command/executable)
my_job['Executable'] = 'job.sh'

# To import files from remote sites to the job’s working directory
my_job['Imports'] = [{
    "From": "https://raw.githubusercontent.com/the-virtual-brain/tvb-multiscale/update-collab-examples/docker/cosimulate_tvb_nest.sh",
    "To" : job.sh
}]

# Specify the resources to request on the remote system
my_job['Resources'] = {  
    "CPUs": "1"}
```

### 3. Actual job submission
In order to submit a job we have to use the JSON built in the previous step and also if we have some local files, we have to give their paths as a list of strings (inputs argument) so the UNICORE library will upload them in the job's working directory in the staging in phase, before launching the job.
```sh
job = site_client.new_job(job_description=my_job, inputs=['/path1', '/path2'])
job.properties
```

### 4. Wait until job is completed and check the results
Wait until the job is completed using the following command
```sh
# TRUE or FALSE
job.is_running()
```
Check job's working directory for the output files/directories using
```sh
wd = job.working_dir
wd.listdir()
```
From working job you can preview files content and download files
```sh
# Read 'stdout' file
out = wd.stat("stdout")
f = out.raw()
all_lines = f.read().splitlines()
all_lines[-20:]
```

```sh
# Download 'outputs/res/results.npy' file
wd.stat("outputs/res/results.npy").download("results.npy")
```