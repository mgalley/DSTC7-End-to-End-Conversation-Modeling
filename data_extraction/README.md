# Data Extraction for DSTC7: End-to-End Conversation Modeling 

Task 2 uses conversational data extracted from Reddit, along with the text of the link that started these conversations. This page provides scripts to extract the data from a Reddit [dump](http://files.pushshift.io/reddit/comments/), as we are unable to release the data directly ourselves.

Note: In the original proposal, we planned to use Twitter data (conversational data) and Foursquare (grounded data), but decided to use Reddit, owing to the volatility of Twitter data, as well the technical difficulties of aligning Twitter content with data from other sources.  Reddit provides an intuitive direct link to external data in the submissions that can be utilized for this task. More details will be provided in a forthcoming update.

## Requirements
Python 3.x
nltk
beautifulsoup4

## How to use
To create the trial data, please run src/create_trial_data.sh. Details will be available shortly (Note: thiw script crawls data from the web, but it reads and respect robots.txt files).
