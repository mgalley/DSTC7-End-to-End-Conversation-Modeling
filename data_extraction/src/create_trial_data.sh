#!/bin/bash

mkdir reddit
mkdir data
mkdir logs

for id in `cat lists/reddit.txt`; do
	echo Downloading $id
	wget https://files.pushshift.io/reddit/submissions/RS_$id.bz2 -O reddit/RS_$id.bz2 -o reddit/RS_$id.bz2.log -c
	wget https://files.pushshift.io/reddit/comments/RC_$id.bz2 -O reddit/RC_$id.bz2 -o reddit/RC_$id.bz2.log -c
	python src/create_trial_data.py --rsinput=reddit/RS_$id.bz2 --rcinput=reddit/RC_$id.bz2 --subreddit_filter=lists/subreddits.txt --domain_filter=lists/domains.txt --pickle=data/$id.pkl --facts=data/$id.facts.txt --convos=data/$id.convos.txt > logs/$id.log 2> logs/$id.err
done

