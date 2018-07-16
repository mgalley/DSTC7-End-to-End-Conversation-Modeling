.SECONDARY:

all: train dev

dev: data-official/dev.convos.txt data-official/dev.facts.txt
train: data-official/train.convos.txt data-official/train.facts.txt

include src/Makefile.official.targets

data-official/dev.convos.txt: $(OFFICIAL_DEV_CONVOS)
	cat $+ > $@

data-official/dev.facts.txt: $(OFFICIAL_DEV_FACTS)
	cat $+ > $@

data-official/train.convos.txt: $(OFFICIAL_TRAIN_CONVOS)
	cat $+ > $@

data-official/train.facts.txt: $(OFFICIAL_TRAIN_FACTS)
	cat $+ > $@

data-official/%.facts.txt data-official/%.convos.txt: reddit/RC_%.bz2 reddit/RS_%.bz2 data-official/.create
	python src/create_official_data.py --rsinput=reddit/RS_$(*F).bz2 --rcinput=reddit/RC_$(*F).bz2 --subreddit_filter=lists/subreddits-official.txt --domain_filter=lists/domains-official.txt --pickle=data-official/$(*F).pkl --facts=data-official/$(*F).facts.txt --convos=data-official/$(*F).convos.txt --anchoronly=True --use_cc=True > logs/$(*F).log 2> logs/$(*F).err

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
