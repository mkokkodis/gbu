#!/bin/bash
COUNTER=0

#########################################################
########## HMM Sahoo  ########################################
#########################################################

echo "Starting"
algorithm=sahoo
for states in 5 6; do
  for fold in $(seq 0 9); do
    COUNTER=$(($COUNTER + 1))
    qsub -F " $states $fold" pbs/sahoo
  done
done

########################################################
######### HMM  ########################################
#########################################################

algorithm=hmm
for states in 5 6; do
  for fold in $(seq 0 9); do
    COUNTER=$(($COUNTER + 1))
    qsub -F "$states $fold " pbs/hmm
  done
done

############################################################
############# RF ########################################
############################################################

algorithm=rf
for fold in $(seq 0 9); do
  for max_depth in 3 10 15; do
    for n_estimators in 10 50 100; do
      COUNTER=$(($COUNTER + 1))
      qsub -F "$fold $max_depth $n_estimators" pbs/rf
    done
  done
done

###########################################################
############ LSTM ########################################
###########################################################
#
algorithm=lstm
for fold in $(seq 0 9); do
  for epochs in 10 20 30; do
    for batch_size in 32 64 128; do
      COUNTER=$(($COUNTER + 2))
      qsub -F "$fold  $epochs $batch_size" pbs/lstm_C
      qsub -F "$fold  $epochs $batch_size" pbs/lstm
    done
  done
done

###
############################################################
############# XG ########################################
############################################################

algorithm=xg
for fold in $(seq 0 9); do
  for max_depth in 3 10 15; do
    for n_estimators in 50 150 100; do
      for subsample in 0.8 1; do
        COUNTER=$(($COUNTER + 1))
        qsub -F "$fold $max_depth $n_estimators $subsample" pbs/xg
      done
    done
  done
done

echo $COUNTER
