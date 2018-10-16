
# Automatic evaluation script for DSTC7 Task 2

Steps:
1) Make sure you 'git pull' the latest changes (from October 15, 2018), including changes in ../../data_extraction.
2) cd to `../../data_extraction` and type make. This will create the multi-reference file used by the metrics (`../../data_extraction/test.refs`).
3) Install 3rd party software as instructed below (METEOR and mteval-v14c.pl).
5) Run the following command, where `[SUBMISSION]` is the submission file you want to evaluate: (same format as the one you submitted on Oct 8.)
```
python dstc.py -c [SUBMISSION] --refs ../../data_extraction/test.refs
```

Important: the results printed by dstc.py might differ slightly from the official results, if part of your test set failed to download.



# What does it do?
(Based on this [repo](https://github.com/golsun/NLP-tools) by [Sean Xiang Gao](https://www.linkedin.com/in/gxiang1228/))

*  **evaluation**: calculate automated NLP metrics (BLEU, NIST, METEOR, entropy, etc...)
```python
from metrics import nlp_metrics
nist, bleu, meteor, entropy, diversity, avg_len = nlp_metrics(
	  path_refs=["demo/ref0.txt", "demo/ref1.txt"], 
	  path_hyp="demo/hyp.txt")
	  
# nist = [1.8338, 2.0838, 2.1949, 2.1949]
# bleu = [0.4667, 0.441, 0.4017, 0.3224]
# meteor = 0.2832
# entropy = [2.5232, 2.4849, 2.1972, 1.7918]
# diversity = [0.8667, 1.000]
# avg_len = 5.0000
```
* **tokenization**: clean string and deal with punctation, contraction, url, mention, tag, etc
```python
from tokenizers import clean_str
s = " I don't know:). how about this?https://github.com"
clean_str(s)

# i do n't know :) . how about this ? __url__
```

# Requirements
* Works fine for both Python 2.7 and 3.6
* Please **downloads** the following 3rd-party packages and save in a new folder `3rdparty`:
	* [**mteval-v14c.pl**](https://goo.gl/YUFajQ) to compute [NIST](http://www.mt-archive.info/HLT-2002-Doddington.pdf). You may need to install the following [perl](https://www.perl.org/get.html) modules (e.g. by `cpan install`): XML:Twig, Sort:Naturally and String:Util.
	* [**meteor-1.5**](http://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz) to compute [METEOR](http://www.cs.cmu.edu/~alavie/METEOR/index.html). It requires [Java](https://www.java.com/en/download/help/download_options.xml).

