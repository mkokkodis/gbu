##!/bin/bash
algorithm=logit
    for fold in $(seq 0 9); do
  python train.py -a $algorithm -o -f $fold
	done

algorithm=svm
for fold in $(seq 0 9); do
  python train.py -a $algorithm -o -f $fold
done

