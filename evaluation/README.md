# Evaluation

## Evaluation results

* [Spreadsheet](https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling/blob/master/evaluation/dstc7-task2-scores.xlsx) with overall scores for automatic (BLEU, METEOR, and NIST) and human evaluation for each team (anonymized).
* [Spreadsheet](https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling/blob/master/evaluation/dstc7-task2-individual_judgments.xlsx) containing judgments (averaged over 3 crowdworkers) of relevance and interest for each response of each system and baseline (test set of 1k responses). Note that this document is currently only useful to DSTC7 Task2 participants, as it maps a sentence ID (hash) to the responses' numerical scores. The responses themselves are not provided at the moment.

## Important Dates
* 9/10/2018: Task organizers release test data
* 10/1/2018 (11:59pm PST): Deadline for sending system outputs to task organizers (email to <dstc7-task2@microsoft.com>)
* 10/8/2018: Task organizers release automatic evaluation results (BLEU, METEOR, etc.)
* 10/16/2018: Task organizers release human evaluation results
* 11/5/2018: System descriptions due to [DSTC7](http://workshop.colips.org/dstc7/) organizers (coming soon)

## Create test data (estimated time: 1 week maximum)

Please refer to the [data extraction page](https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling/tree/master/data_extraction) to create data. The scripts on that page have just been updated to create validation and test data, so please dowload the latest version of the code (e.g., with `git pull`). Then, you just need to run the following command:

```make -j4 valid test```

This will create the followng four files:

* Validation data: ``valid.convos.txt`` and ``valid.facts.txt``
* Test data: ``test.convos.txt`` and ``test.facts.txt``

These files are in exactly the same format as ``train.convos.txt`` and ``train.facts.txt`` already explained [here](https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling/tree/master/data_extraction). The only difference is that the ``response`` field of test.convos.txt has been replaced with the strings ``__UNDISCLOSED__``.

Notes: 
* The two validation files are optional and you can skip them if you want (e.g., no need to send us system outputs for them). We provide them so that you can run your own automatic evaluation (BLEU, etc.) by comparing the ``response`` field with your own system outputs. 
* Obviously, you may not use the validation or test data for training any of your models or systems.
* Data creation should take about 1-4 days (depending on your internet connection, etc.). If you run into trouble creating the data or data extraction isn't complete by September 17, please contact us.

### Data statistics

Number of conversational responses: 
* Validation (valid.convos.txt): 4542 lines
* Test (test.convos.txt): 13440 lines

Due to the way the data is created by querying Common Crawl, there may be small differences between your version of the data and our own. To make pairwise comparisons between systems of each pair of participants, we will rely on the largest subset of the test set that is common to both participants.  **However, if your file test.convos.txt contains less than 13,000 lines, this might be an indication of a problem so please contact us immediately**.

## Create system outputs (estimated time: 2 weeks maximum)

**By October 8th**, please send us a modification of ``test.convos.txt`` where ``__UNDISCLOSED__`` has been replaced by your own system output, which is an output based on the query, subreddit, and conversation ID specified on the same line. On October 4th, we sent an email to registered participants with a URL to the system output submission site. If you registed but didn't receive such an email, please contact us at <dstc7-task2@microsoft.com>.

In order for us to process these files automatically, please ensure the following about file format:
* **Other than replacing ``__UNDISCLOSED__`` with your own output, the rest of the file should not be altered in any way.** That is, it needs to have exactly 7 columns on each line, and columns 1-6 should be exactly the same as the test.convos.txt file created by our scripts. These other columns are also important as they will help us sort out differences between the test sets of the different participants.
* You may submit multiple systems, in which case we ask that you: (1) Please give a different name to each file (e.g., system1.convos.txt, system2.convos.txt, ...). (2) Please identify one of your system as primary, as we may only be able to use one of them for human evaluation.
* Please do not send us any other files (e.g., we do not need validation sets, and any facts files.) 

*Before submitting, **please check the format of your files** and make sure they are as described above, as we will process submissions automatically for metric evaluation (and semi-automatically for human evaluation). Formatting your data correctly as described above will minimize problems in the evaluation pipeline and the risk of getting lower metric scores. If you are not sure about the format, just ask.*

If you have any questions, again please email us at <dstc7-task2@microsoft.com>.
