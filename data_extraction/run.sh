mkdir -p logs
mkdir -p data-official-valid
touch data-official-valid/.create
python src/create_official_data.py --rsinput=reddit/RS_2017-01.bz2 --rcinput=reddit/RC_2017-01.bz2 --subreddit_filter=lists/subreddits-official.txt --domain_filter=lists/domains-official.txt --pickle=data-official-valid/2017-01.pkl --facts=data-official-valid/2017-01.facts.txt --convos=data-official-valid/2017-01.convos.txt --anchoronly=True --use_cc=True --test=lists/valid.hashes > logs/2017-01.log 2> logs/2017-01.err
