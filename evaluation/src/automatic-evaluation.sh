#!/bin/bash

# Automatic evaluation script:

# Name of any of the files submitted to DSTC7 Task2
# (or any more recent file)
SUBMISSION=systems/constant-baseline.txt

# Make sure this file exists:
REFS=../../data_extraction/test.refs

if [ ! -f $REFS ]; then
	echo "Reference file not found. Please move to ../../data_extraction and type make."
else
	python dstc.py -c $SUBMISSION --refs $REFS
fi

