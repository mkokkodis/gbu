#!/bin/bash

for a in svm logit; do
  cat ../../data/local_results_per_model/$a/* > ../../data/results/$a.csv
done