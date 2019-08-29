#!/bin/bash -e

echo "removing unneeded files...."
rm -f tune_*/*.job tune_*/Makefile tune_*/*_exe? tune_*/*_part*.cu tune_*/*_part*.cpp tune_*/*.o

fn="../libsmm_acc_tuning_`date +'%F'`.tgz"

if [ -f $fn ]; then
   echo "Archive file exists already, aborting!"
   exit 1
fi

set -x
tar czf $fn .
