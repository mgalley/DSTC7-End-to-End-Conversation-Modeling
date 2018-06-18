# Data Extraction for DSTC7: End-to-End Conversation Modeling 

Task 2 uses conversational data extracted from Reddit, along with the text of the link that started these conversations. This page provides scripts to extract the data from a Reddit [dump](http://files.pushshift.io/reddit/comments/), as we are unable to release the data directly ourselves.

*Note: In the original proposal, we planned to use Twitter data (conversational data) and Foursquare (grounded data), but decided to use Reddit, owing to the volatility of Twitter data, as well the technical difficulties of aligning Twitter content with data from other sources. Reddit provides an intuitive direct link to external data in the submissions that can be utilized for this task.*

## Requirements

* `Python 3.x`, with modules:
   * `nltk`
   * `beautifulsoup4`
* `make`
* `wget`


## Create trial data:

To create the trial data, please run:

```src/create_trial_data.sh```

This will create two tab-separated (tsv) files `data/trial.convos.txt` and `data/trial.facts.txt`, which respectively contain the conversational data and grounded text ("facts"). This requires about 20 GB of disk space.

### Notes:

* **Web crawling**: The above script downloads grounding information directly from the web, but does respect the servers' `robots.txt` rules. The official version of the data (forthcoming) will extract that data from [Common Crawl](http://commoncrawl.org/), to ensure that all participants use exactly the same data, and to minimize the number of dead links.
* **Data split**: The official data will be divided into train/dev/test, but the trial data isn't.
* **Offensive language**: We restricted the data to subreddits that are generally inoffensive. However, even the most "well behaved" subreddits occasionally contain offensive and explicit language, and the trial-version of the data does not attempt to remove it.

## Data format:

### Conversation file:

Each line of `trial.convos.txt` contains a Reddit response and its preceding conversational context. Long conversational contexts are truncated by keeping the last 100 words. The file contains 4 columns:

1. subreddit name
2. conversation ID
3. response score
4. conversational context (input of the model)
5. response (output of the model)

Sample:
TODO

