# ALCF Theta `plasma-python` FRNN Notes

**Author: Rick Zamora (rzamora@anl.gov)**

This document is intended to act as a tutorial for running the [plasma-python](https://github.com/PPPLDeepLearning/plasma-python) implementation of the Fusion recurrent neural network (FRNN) on the ALCF Theta supercomputer (Cray XC40; Intel KNL processors).  The steps followed in these notes are based on the Princeton [Tiger-GPU tutorial](https://github.com/PPPLDeepLearning/plasma-python/blob/master/docs/PrincetonUTutorial.md#location-of-the-data-on-tigress), hosted within the main GitHub repository for the project.

## Environment Setup


Choose a *root* directory for FRNN-related installations on Theta:

```
export FRNN_ROOT=<desired-root-directory>
cd $FRNN_ROOT
```

Create a simple directory structure allowing experimental *builds* of the `plasma-python` python code/library:

```
mkdir build
mkdir build/miniconda-3.6-4.5.4
cd build/miniconda-3.6-4.5.4
```

### Custom Miniconda Environment Setup

Copy miniconda installation script to working directory (and install):

```
cp /lus/theta-fs0/projects/fusiondl_aesp/rzamora/scripts/install_miniconda-3.6-4.5.4.sh .
./install_miniconda-3.6-4.5.4.sh
```

The `install_miniconda-3.6-4.5.4.sh` script will install `miniconda-4.5.4` (using `Python-3.6`), as well as `Tensorflow-1.12.0` and `Keras 2.2.4`.


Update your environment variables to use miniconda:

```
export PATH=${FRNN_ROOT}/build/miniconda-3.6-4.5.4/miniconda3/4.5.4/bin:$PATH
export PYTHONPATH=${FRNN_ROOT}/build/miniconda-3.6-4.5.4/miniconda3/4.5.4/lib/python3.6/site-packages/:$PYTHONPATH
```

Note that the previous lines (as well as the definition of `FRNN_ROOT`) can be appended to your `$HOME/.bashrc` file if you want to use this environment on Theta by default.


## Installing `plasma-python`

Here, we assume the installation is within the custom miniconda environment installed in the previous steps. We also assume the following commands have already been executed:

```
export FRNN_ROOT=<desired-root-directory>
export PATH=${FRNN_ROOT}/build/miniconda-3.6-4.5.4/miniconda3/4.5.4/bin:$PATH
export PYTHONPATH=${FRNN_ROOT}/build/miniconda-3.6-4.5.4/miniconda3/4.5.4/lib/python3.6/site-packages/:$PYTHONPATH
```

*Personal Note: Using `export FRNN_ROOT=/home/zamora/ESP/FRNN_project`*

If the environment is set up correctly, installation of `plasma-python` is straightforward:

```
cd ${FRNN_ROOT}/build/miniconda-3.6-4.5.4
git clone https://github.com/PPPLDeepLearning/plasma-python.git
cd plasma-python
python setup.py build
python setup.py install
```

## Data Access

Sample data and metadata is available in `/lus/theta-fs0/projects/fusiondl_aesp/FRNN/tigress/alexeys/signal_data` and `/lus/theta-fs0/projects/fusiondl_aesp/FRNN/tigress/alexeys/shot_lists`, respectively.  It is recommended that users create their own symbolic links to these directories. I recommend that you do this within a directory called `/lus/theta-fs0/projects/fusiondl_aesp/<your-alcf-username>/`. For example:

```
ln -s /lus/theta-fs0/projects/fusiondl_aesp/FRNN/tigress/alexeys/shot_lists  /lus/theta-fs0/projects/fusiondl_aesp/<your-alcf-username>/shot_lists
ln -s /lus/theta-fs0/projects/fusiondl_aesp/FRNN/tigress/alexeys/signal_data  /lus/theta-fs0/projects/fusiondl_aesp/<your-alcf-username>/signal_data
```

For the examples included in `plasma-python`, there is a configuration file that specifies the root directory of the raw data. Change the `fs_path: '/tigress'` line in `examples/conf.yaml` to reflect the following:

```
fs_path: '/lus/theta-fs0/projects/fusiondl_aesp'
```

Its also a good idea to change `num_gpus: 4` to `num_gpus: 1`. I am also using the `jet_data_0D` dataset:

```
paths:
    data: jet_data_0D
```


### Data Preprocessing

#### The SLOW Way (On Theta)

Theta is KNL-based, and is **not** the best resource for processing many text files in python. However, the preprocessing step *can* be used by using the following steps (although it may need to be repeated many times to get through the whole dataset in a 60-minute debug queues):

```
cd ${FRNN_ROOT}/build/miniconda-3.6-4.5.4/plasma-python/examples
cp /lus/theta-fs0/projects/fusiondl_aesp/FRNN/rzamora/scripts/submit_guarantee_preprocessed.sh .
```

Modify the paths defined in `submit_guarantee_preprocessed.sh` to match your environment.

Note that the preprocessing module will use Pathos multiprocessing (not MPI/mpi4py).  Therefore, the script will see every compute core (all 256 per node) as an available resource.  Since the LUSTRE file system is unlikely to perform well with 256 processes (on the same node) opening/closing/creating files at once, it might improve performance if you make a slight change to line 85 in the `vi ~/plasma-python/plasma/preprocessor/preprocess.py` file:

```
line 85: use_cores = min( <desired-maximum-process-count>, max(1,mp.cpu_count()-2) )
```

After optionally re-building and installing plasm-python with this change, submit the preprocessing job:

```
qsub submit_guarantee_preprocessed.sh
```

#### The FAST Way (On Cooley)

You will fine it much less painful to preprocess the data on Cooley, because the Haswell processors are much better suited for this... Log onto the ALCF Cooley Machine:

```
ssh <alcf-username>@cooley.alcf.anl.gov
```

Copy my `cooley_preprocess` example directory to whatever directory you choose to work in:

```
cp /lus/theta-fs0/projects/fusiondl_aesp/FRNN/rzamora/scripts/cooley_preprocess .
cd cooley_preprocess
```

This directory has a Singularity image with everything you need to run your code on Cooley. Assuming you have created symbolic links to the `shot_lists` and `signal_data` directories in `/lus/theta-fs0/projects/fusiondl_aesp/<your-alcf-username>/`, you can just submit the included `COBALT` script (to specify the data you want to process, just modify the included `conf.yaml` file):

```
qsub submit.sh
```

For me, this finishes in less than 10 minutes, and creates 5523 `.npz` files in the `/lus/theta-fs0/projects/fusiondl_aesp/<your-alcf-username>/processed_shots/` directory.  The output file of the COBALT submission ends with the following message:

```
5522/5523Finished Preprocessing 5523 files in 406.94421911239624 seconds
Omitted 5523 shots of 5523 total.
0/0 disruptive shots
WARNING: All shots were omitted, please ensure raw data is complete and available at /lus/theta-fs0/projects/fusiondl_aesp/zamora/signal_data/.
4327 1196
```






