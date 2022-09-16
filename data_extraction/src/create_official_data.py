#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
create_official_data.py
A script to turn extract:
(1) conversation from Reddit file dumps (originally downloaded from https://files.pushshift.io/reddit/daily/)
(2) grounded data ("facts") extracted from the web, respecting robots.txt
"""

import sys
import io
import time
import os.path
import re
import argparse
import traceback
import json
import zstandard as zstd
import pickle
import nltk
import urllib.request
import urllib.robotparser
import hashlib

from bs4 import BeautifulSoup
from bs4.element import NavigableString
from bs4.element import CData
from multiprocessing import Pool
from nltk.tokenize import TweetTokenizer
from commoncrawl import CommonCrawl

parser = argparse.ArgumentParser()
parser.add_argument("--rsinput", help="Submission (RS) file to load.")
parser.add_argument("--rcinput", help="Comments (RC) file to load.")
parser.add_argument("--test", help="Hashes of test set convos.", default="")
parser.add_argument("--facts", help="Facts file to create.")
parser.add_argument("--convos", help="Convo file to create.")
parser.add_argument("--pickle", help="Pickle that contains conversations and facts.", default="data.pkl")
parser.add_argument("--subreddit_filter", help="List of subreddits (inoffensive, safe for work, etc.)")
parser.add_argument("--domain_filter", help="Filter on subreddits and domains.")
parser.add_argument("--nsubmissions", help="Number of submissions to process (< 0 means all)", default=-1, type=int)
parser.add_argument("--min_fact_len", help="Minimum number of tokens in each fact (reduce noise in html).", default=0, type=int)
parser.add_argument("--min_res_len", help="Min number of characters in response.", default=2, type=int)
parser.add_argument("--max_res_len", help="Max number of characters in response.", default=280, type=int)
parser.add_argument("--max_context_len", help="Max number of words in context.", default=200, type=int)
parser.add_argument("--max_depth", help="Maximum length of conversation.", default=5, type=int)
parser.add_argument("--mincomments", help="Minimum number of comments per submission.", default=10, type=int)
parser.add_argument("--minscore", help="Minimum score of each comment.", default=1, type=int)
parser.add_argument("--delay", help="Seconds of delay when crawling web pages", default=0, type=int)
parser.add_argument("--tokenize", help="Whether to tokenize facts and conversations.", default=True, type=bool)
parser.add_argument("--anchoronly", help="Filter out URLs with no named anchors.", default=False, type=bool)
parser.add_argument("--use_robots_txt", help="Whether to respect robots.txt (disable this only if urls have been previously checked by other means!)", default=True, type=bool)
parser.add_argument("--use_cc", help="Whether to download pages from Common Crawl.", default=False, type=bool)
parser.add_argument("--cc_match", help="tsv file containing the precomputation of [url, submission month] -> [cc month]", default='lists/cc-match.tsv', type=str)
parser.add_argument("--dryrun", help="Just collect stats about data; don't create any data.", default=False, type=bool)
parser.add_argument("--blind", help="Don't print out responses.", default=False, type=bool)
args = parser.parse_args()

fields = [ "id", "subreddit", "score", "num_comments", "domain", "title", "url", "permalink" ]
important_tags = ['title', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'hr', 'p']
notext_tags = ['script', 'style']
deleted_str = '[deleted]'
undisclosed_str = '__UNDISCLOSED__'
max_window_size = 2**31

batch_download_facts = False
robotparsers = {}
tokenizer = TweetTokenizer(preserve_case=False)
if args.cc_match:
    cc = CommonCrawl(0)
else:
    cc = CommonCrawl(-2)

def get_subreddit(submission):
    return submission["subreddit"]
def get_domain(submission):
    return submission["domain"]
def get_url(submission):
    return submission["url"]
def get_submission_text(submission):
    return submission["title"]
def get_permalink(submission):
    return submission["permalink"]
def get_submission_id(submission):
    return submission["id"]
def get_comment_id(comment):
    return "t1_" + comment["id"]
    #return comment["name"]
def get_parent_comment_id(comment):
    return comment["parent_id"]
def get_text(comment):
    return comment["body"]
def get_user(comment):
    return comment["author"]
def get_score(comment):
    return comment["score"]
def get_linked_submission_id(comment):
    return comment["link_id"].split("_")[1]

def get_anchor(url):
    pos = url.find("#")
    if (pos > 0):
        label = url[pos+1:]
        label = label.strip()
        return label
    return ""

def filter_submission(submission):
    """Determines whether to filter out this submission (over-18, deleted user, etc.)."""
    if submission["num_comments"] < args.mincomments:
        return True
    if "num_crossposts" in submission and submission["num_crossposts"] > 0:
        return True
    if "locked" in submission and submission["locked"]:
        return True
    if "over-18" in submission and submission["over_18"]:
        return True
    if "brand_safe" in submission and not submission["brand_safe"]:
        return True
    if submission["distinguished"] != None:
        return True
    if "subreddit_type" in submission:
        if submission["subreddit_type"] == "restricted": # filter only public
            return True
        if submission["subreddit_type"] == "archived":
            return True
    url = get_url(submission)
    domain = get_domain(submission)
    if domain.find("reddit.com") >= 0 or domain.find("twitter.com") >= 0 or domain.find("youtube.com") >= 0 or domain.find("youtube.com") >= 0 or domain.find("imgur.com") >= 0 or domain.find("flickr.com") >= 0 or domain.find("ebay.com") >= 0:
        return True
    if args.anchoronly and len(get_anchor(url)) <= 2:
        return True
    if url.find(" ") >= 0:
        return True
    if url.endswith("jpg") or url.endswith("gif") or url.endswith("png") or url.endswith("pdf"):
        return True
    return False

def norm_article(t):
    """Minimalistic processing with linebreaking."""
    t = re.sub("\s*\n+\s*","\n", t)
    t = re.sub(r'(</[pP]>)',r'\1\n', t)
    t = re.sub("[ \t]+"," ", t)
    t = t.strip()
    return t

def norm_sentence(t):
    """Minimalistic processing: remove extra space characters."""
    t = re.sub("[ \n\r\t]+", " ", t)
    t = t.strip()
    if args.tokenize:
        t = " ".join(tokenizer.tokenize(t))
        t = t.replace('[ deleted ]','[deleted]');
    return t

def add_webpage(submission, year, month, cc_match=None):
    """Retrive sentences ('facts') from submission["url"]. """
    if args.use_cc:
        return add_cc_webpage(submission, year, month, cc_match)
    return add_live_webpage(submission)

def add_cc_webpage(submission, year, month, cc_match=None):
    url = get_url(submission)
    kstr = cc.get_key(url, month, year)
    if cc_match and kstr not in cc_match:
        print("Can't fetch (precomputed): [%s] submission month: [%s-%s]" % (url, year, month))
        sys.stdout.flush()
        return None
    src, month_id, date = cc.download(url, year, month, False, cc_match=cc_match)
    sys.stdout.flush()
    if src == None:
        src, month_id, date = cc.download(url, year, month, True, cc_match=cc_match)
        sys.stdout.flush()
        if src == None:
            print("Can't fetch: [%s] submission month: [%s-%s]" % (url, year, month))
            sys.stdout.flush()
            return None
    print("Fetching url: [%s] submission month: [%s-%s] -> [%s] commoncrawl date: [%s]" % (url, year, month, month_id, str(date)))
    print("FETCH\t%s\t%s-%s\t%s" % (url, year, month, month_id))
    sys.stdout.flush()
    submission["source"] = src
    return submission

def add_live_webpage(submission):
    url = get_url(submission)
    domain = get_domain(submission)
    try:
        if args.use_robots_txt:
            if args.delay > 0:
                time.sleep(args.delay)
            if domain in robotparsers.keys():
                rp = robotparsers[domain]
            else:
                rp = urllib.robotparser.RobotFileParser()
                robotparsers[domain] = rp
                rurl = "http://" + domain + "/robots.txt"
                print("Fetching robots.txt: [%s]" % rurl)
                rp.set_url(rurl)
                rp.read()
            if not rp.can_fetch("*", url):
                print("Can't download url due to robots.txt: [%s] domain: [%s]" % (url, domain))
                return None
        print("Fetching url: [%s] domain: [%s]" % (url, domain))
        u = urllib.request.urlopen(url)
        src = u.read()
        submission["source"] = src
        return submission
    except urllib.error.HTTPError:
        return None
    except urllib.error.URLError:
        return None
    except UnicodeEncodeError:
        return None
    except:
        traceback.print_exc()
        return None

def add_webpages(submissions):
    """Use multithreading to retrieve multiple webpages at once."""
    print("Downloading %d pages:" % len(submissions))
    pool = Pool()
    submissions = pool.map(add_webpage, submissions)
    print("\nDone.")
    return [s for s in submissions if s is not None]

def get_date(file_name):
    m = re.search(r'(\d\d\d\d)-(\d\d)', file_name)
    year = m.group(1)
    month = m.group(2)
    return year, month

def get_submissions(rs_file, subreddit_file, domain_file, cc_match=None):
    """Return all submissions from a dump submission file rs_file (RS_*.zst),
       restricted to the subreddit+domain listed in filter_file."""
    submissions = []
    subreddit_dic = None
    domain_dic = None
    year, month = get_date(rs_file)
    if subreddit_file != None:
        with open(subreddit_file) as f:
            subreddit_dic = dict([ (el.strip(), 1) for el in f.readlines() ])
    if domain_file != None:
        with open(domain_file) as f:
            domain_dic = dict([ (el.strip(), 1) for el in f.readlines() ])

    with open(rs_file, 'rb') as fh:
        dctx = zstd.ZstdDecompressor(max_window_size=max_window_size)
        with dctx.stream_reader(fh) as reader:
            i = 0
            for line in io.TextIOWrapper(io.BufferedReader(reader), encoding='utf-8'):
                try:
                    submission = json.loads(line)
                    if not filter_submission(submission):
                        subreddit = get_subreddit(submission)
                        domain = get_domain(submission)
                        scheck = subreddit_dic == None or subreddit in subreddit_dic
                        dcheck = domain_dic == None or domain in domain_dic
                        if scheck and dcheck:
                            s = dict([ (f, submission[f]) for f in fields ])
                            print("keeping: subreddit=%s\tdomain=%s" % (subreddit, domain))
                            if args.dryrun:
                                continue
                            if not batch_download_facts:
                                s = add_webpage(s, year, month, cc_match=cc_match)
                            submissions.append(s)
                            sys.stdout.flush()
                            sys.stderr.flush()
                            i += 1
                            if i == args.nsubmissions:
                                break
                        else:
                            print("skipping: subreddit=%s\tdomain=%s (%s %s)" % (subreddit, domain, scheck, dcheck))
                            pass
                except json.decoder.JSONDecodeError:
                    pass
                except Exception:
                    traceback.print_exc()
                    pass
    if batch_download_facts:
        submissions = add_webpages(submissions)
    else:
        submissions = [s for s in submissions if s is not None]
    return dict([ (get_submission_id(s), s) for s in submissions ])

def get_comments(rc_file, submissions):
    """Return all conversation triples from rc_file (RC_*.bz2),
       restricted to given submissions."""
    comments = {}
    with open(rc_file, 'rb') as fh:
        dctx = zstd.ZstdDecompressor(max_window_size=max_window_size)
        with dctx.stream_reader(fh) as reader:
            for line in io.TextIOWrapper(io.BufferedReader(reader), encoding='utf-8'):
                try:
                    comment = json.loads(line)
                    sid = get_linked_submission_id(comment)
                    if sid in submissions.keys():
                        comments[get_comment_id(comment)] = comment
                except Exception:
                    traceback.print_exc()
                    pass
    return comments

def load_data(cc_match=None):
    """Load data either from a pickle file if it exists,
       and otherwise from RC_* RS_* and directly from the web."""
    if not os.path.isfile(args.pickle):
        submissions = get_submissions(args.rsinput, args.subreddit_filter, args.domain_filter, cc_match=cc_match)
        comments = get_comments(args.rcinput, submissions)
        with open(args.pickle, 'wb') as f:
            pickle.dump([submissions, comments], f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(args.pickle, 'rb') as f:
            [submissions, comments] = pickle.load(f)
    return submissions, comments

def insert_escaped_tags(tags, label=None):
    """For each tag in "tags", insert contextual tags (e.g., <p> </p>) as escaped text
       so that these tags are still there when html markup is stripped out."""
    found = False
    for tag in tags:
        strs = list(tag.strings)
        if len(strs) > 0:
            if label != None:
                l = label
            else:
                l = tag.name
            strs[0].parent.insert(0, NavigableString("<"+l+">"))
            strs[-1].parent.append(NavigableString("</"+l+">"))
            found = True
    return found

def save_facts(submissions, sids = None):
    subs = {}
    i = 0
    if args.facts == '-':
        return submissions
    with open(args.facts, 'wt', encoding="utf-8") as f:
        for id in sorted(submissions.keys()):
            s = submissions[id]
            url = get_url(s)
            label = get_anchor(url)
            print("Processing submission %s...\n\turl: %s\n\tanchor: %s\n\tpermalink: http://reddit.com%s" % (id, url, str(label), get_permalink(s)))
            subs[id] = s
            if sids == None or id in sids.keys():
                b = BeautifulSoup(s["source"],'html.parser')
                # If there is any anchor in the url, locate it in the facts:
                if label != "":
                    if not insert_escaped_tags(b.find_all(True, attrs={"id": label}), 'anchor'):
                        print("\t(couldn't find anchor on page: %s)" % label)
                # Remove tags whose text we don't care about (javascript, etc.):
                for el in b(notext_tags):
                    el.decompose()
                # Delete other unimportant tags, but keep the text:
                for tag in b.findAll(True):
                    if tag.name not in important_tags:
                        tag.append(' ')
                        tag.replaceWithChildren()
                # All tags left are important (e.g., <p>) so add them to the text: 
                insert_escaped_tags(b.find_all(True))
                # Extract facts from html:
                t = b.get_text(" ")
                t = norm_article(t)
                facts = []
                for sent in filter(None, t.split("\n")):
                    if len(sent.split(" ")) >= args.min_fact_len:
                       facts.append(norm_sentence(sent))
                for fact in facts:
                    out_str = "\t".join([get_subreddit(s), id, get_domain(s), fact])
                    hash_str = hashlib.sha224(out_str.encode("utf-8")).hexdigest()
                    f.write(hash_str + "\t" + out_str + "\n")
                s["facts"] = facts
                i += 1
                if i == args.nsubmissions:
                    break
    return subs

def get_convo(id, submissions, comments, depth=args.max_depth):
    c = comments[id]
    pid = get_parent_comment_id(c)
    if pid in comments.keys() and depth > 0:
        els = get_convo(pid, submissions, comments, depth-1)
    else:
        s = submissions[get_linked_submission_id(c)]
        els = [ "START", norm_sentence(get_submission_text(s)) ]
    els.append(norm_sentence(get_text(c)))
    return els

def save_tuple(f, subreddit, sid, pos, user, context, message, response, score, test_hashes):
    cwords = re.split("\s+", context)
    mwords = re.split("\s+", message)
    max_len = max(args.max_context_len, len(mwords)+1)
    if len(cwords) > max_len:
        ndel = len(cwords) - max_len
        del cwords[:ndel]
        context = "... " + " ".join(cwords)
    if len(response) <= args.max_res_len and len(response) >= args.min_res_len and response != deleted_str and user != deleted_str and response.find(">") < 0:
        if context.find(deleted_str) < 0:
            if score >= args.minscore:
                out_str = "\t".join([subreddit, sid, str(score), str(pos), context, response])
                hash_str = hashlib.sha224(out_str.encode("utf-8")).hexdigest()
                if test_hashes == None or hash_str in test_hashes.keys():
                    if args.blind:
                        ## Note: there is no point in removing the '--blind' flag in order to peek at the reference responses (gold), 
                        ## as the organizers will rely on different responses to compute BLEU, etc.
                        out_str = "\t".join([subreddit, sid, str(score), str(pos), context, undisclosed_str])
                    f.write(hash_str + "\t" + out_str + "\n")

def save_tuples(submissions, comments, test_hashes):
    has_firstturn = {}
    with open(args.convos, 'wt', encoding="utf-8") as f:
        for id in sorted(comments.keys()):
            comment = comments[id]
            user = get_user(comment)
            score = get_score(comment)
            sid = get_linked_submission_id(comment)
            if sid in submissions.keys():
                s = submissions[sid]
                convo = get_convo(id, submissions, comments)
                pos = len(convo) - 1
                context = " EOS ".join(convo[:-1])
                message = convo[-2]
                response = convo[-1]
                if len(convo) == 3 and not sid in has_firstturn.keys():
                    save_tuple(f, get_subreddit(s), sid, pos-1, "", convo[-3], convo[-3], message, 1, test_hashes)
                    has_firstturn[sid] = 1
                save_tuple(f, get_subreddit(s), sid, pos, user, context, message, response, score, test_hashes)

def read_test_hashes(hash_file):
    hashes = {}
    with open(hash_file, 'r') as f:
        for line in f:
            hash_str = line.rstrip()
            hashes[hash_str] = 1
    return hashes

def load_cc_match(match_file):
    cc_match = {}
    with open(match_file, 'r') as f:
        for line in f:
            els = line.strip().split('\t')
            k = cc.get_key(els[0], els[1])
            cc_match[k] = els[2]
    return cc_match

if __name__== "__main__":
    test_hashes = None
    if args.test != "":
        test_hashes = read_test_hashes(args.test)
    cc_match = None
    if args.cc_match != '':
        cc_match = load_cc_match(args.cc_match)
    submissions, comments = load_data(cc_match=cc_match)
    submissions = save_facts(submissions)
    save_tuples(submissions, comments, test_hashes)
