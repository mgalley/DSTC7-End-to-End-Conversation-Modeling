.SECONDARY:

all: train dev valid test

test: data-official-test/test.convos.txt data-official-test/test.facts.txt
valid: data-official-valid/valid.convos.txt data-official-valid/valid.facts.txt
dev: data-official/dev.convos.txt data-official/dev.facts.txt
train: data-official/train.convos.txt data-official/train.facts.txt

include src/Makefile.official.targets

data-official-test/test.convos.txt: $(OFFICIAL_TEST_CONVOS)
	cat $+ | sort | uniq > $@

data-official-test/test.facts.txt: $(OFFICIAL_TEST_FACTS)
	cat $+ > $@

data-official-valid/valid.convos.txt: $(OFFICIAL_VALID_CONVOS)
	cat $+ | sort | uniq > $@

data-official-valid/valid.facts.txt: $(OFFICIAL_VALID_FACTS)
	cat $+ > $@

data-official/dev.convos.txt: $(OFFICIAL_DEV_CONVOS)
	cat $+ > $@

data-official/dev.facts.txt: $(OFFICIAL_DEV_FACTS)
	cat $+ > $@

data-official/train.convos.txt: $(OFFICIAL_TRAIN_CONVOS)
	cat $+ > $@

data-official/train.facts.txt: $(OFFICIAL_TRAIN_FACTS)
	cat $+ > $@

## Note: there is no point in removing the '--blind' flag in order to peek at the reference responses (gold), as the organizers will rely on different responses to compute BLEU, etc.
data-official-test/%.facts.txt data-official-test/%.convos.txt: reddit/RC_%.bz2 reddit/RS_%.bz2 data-official-test/.create lists/test.hashes
	python src/create_official_data.py --rsinput=reddit/RS_$(*F).bz2 --rcinput=reddit/RC_$(*F).bz2 --subreddit_filter=lists/subreddits-official.txt --domain_filter=lists/domains-official.txt --pickle=data-official-test/$(*F).pkl --facts=data-official-test/$(*F).facts.txt --convos=data-official-test/$(*F).convos.txt --anchoronly=True --use_cc=True --test=lists/test.hashes --blind=True > logs/$(*F).log 2> logs/$(*F).err

data-official-valid/%.facts.txt data-official-valid/%.convos.txt: reddit/RC_%.bz2 reddit/RS_%.bz2 data-official-valid/.create lists/valid.hashes
	python src/create_official_data.py --rsinput=reddit/RS_$(*F).bz2 --rcinput=reddit/RC_$(*F).bz2 --subreddit_filter=lists/subreddits-official.txt --domain_filter=lists/domains-official.txt --pickle=data-official-valid/$(*F).pkl --facts=data-official-valid/$(*F).facts.txt --convos=data-official-valid/$(*F).convos.txt --anchoronly=True --use_cc=True --test=lists/valid.hashes > logs/$(*F).log 2> logs/$(*F).err

data-official/%.facts.txt data-official/%.convos.txt: reddit/RC_%.bz2 reddit/RS_%.bz2 data-official/.create
	python src/create_official_data.py --rsinput=reddit/RS_$(*F).bz2 --rcinput=reddit/RC_$(*F).bz2 --subreddit_filter=lists/subreddits-official.txt --domain_filter=lists/domains-official.txt --pickle=data-official/$(*F).pkl --facts=data-official/$(*F).facts.txt --convos=data-official/$(*F).convos.txt --anchoronly=True --use_cc=True > logs/$(*F).log 2> logs/$(*F).err

data-official-test/.create:
	mkdir -p logs
	mkdir -p data-official-test
	touch $@

data-official-valid/.create:
	mkdir -p logs
	mkdir -p data-official-valid
	touch $@

data-official/.create:
	mkdir -p logs
	mkdir -p data-official
	touch $@

reddit/RS_%.bz2:
	mkdir -p reddit
	wget https://files.pushshift.io/reddit/submissions/RS_$(*F).bz2 -O reddit/RS_$(*F).bz2 -o logs/RS_$(*F).bz2.log -c

reddit/RC_%.bz2:
	mkdir -p reddit
	wget https://files.pushshift.io/reddit/comments/RC_$(*F).bz2 -O reddit/RC_$(*F).bz2 -o logs/RC_$(*F).bz2.log -c
