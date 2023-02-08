=========The Following work is a draft that may contains some errors typos and draft results also========

***Run the submission_script_alloc_new.sh to simulate the model and train forward direction to  get batch samples Results .
To do so you need to allocate resoursec in the CSCS cluster as following :
to run 500 batch file for each iG value 
500*3=150==>150cpu ,each node contain 72 CPU , we will need 21 node we estimate 14 hours to generate each batch file,
then we are going to use 21*14=294nh of the cluster
===>salloc -A ich012 -J sbi_fit_normal -C mc --nodes=21 --time=14:00:00 --no-shell
once the resources are allocated 
run the shell script
./submission_script_alloc_new.sh 1> output.txt 2> error.txt
<>the output will be bsr_iG0x_yzw.npy
***To make the backward trainng and Test Phase :
Run submission_script_alloc_Valid.sh which calls  sbifit_launcher_valid.sh to run  the python script cwc_FICfit-validation20jan.py .
The submission_script_alloc_Valid.sh needs to be updated for the desired training set number 
A resources allocation is also required each time we run the  submission_script_alloc_Valid.sh as following. 

===>salloc -A ich012 -J sbi_sample_fit -C mc --nodes=3 --time=10:00:00 --no-shell
./submission_script_alloc_Valid.s 1> errorValid.txt 2>OutputValid.txt
<>The output will be samples_fit_iG0x_uyzw_Train.npy 

***After getting the fitting results we need to evaluate this fitting by comparing the generated priors with the expected ones.
And this by calculating the Zscore and the Shrinkage.
to do so we need to run ShrinkagePlotingModified2.py to get shrinkage & ZscorePlotingModified2.py .

prior files , batch sample results and fiting results can be found in the res folder .
figures can be found in the fig folder (many unuseful figures can be found)
