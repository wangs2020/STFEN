#!/bin/bash
#PBS -l nodes=1:ppn=9
#PBS -l walltime=1000:00:00
##PBS -q batch 
#PBS -q new 

#PBS -N resnet_all
##PBS -N informer_twin24
##PBS -N resnet_crossformer_twin24_all
##PBS -N patchtst_twin24

# switch to the directory where the PBS job is submitted
cd $PBS_O_WORKDIR

# this command will run 
##nohup time /public/software/mpi/openmpi/intel/2.1.2/bin/mpiexec -machinefile machines -n 16 ./wrf.sh &

#/public/software/mpi/openmpi/intel/2.1.2/bin/mpirun -n ${nodes} /public/home/wangs/anaconda2/envs/pytorch/bin/python run.py
#/public/software/mpi/openmpi/intel/2.1.2/bin/mpirun -n ${nodes} /public/home/wangs/anaconda2/envs/nc_xarray/bin/python 02extractFromNetcdf.pm.hourly.CPCB.py

/public/home/wangs/anaconda2/envs/pytorch/bin/python train_stfen.py
