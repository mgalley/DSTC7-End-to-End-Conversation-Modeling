# DSTC7: End-to-End Conversation Modeling

**[DSTC7](http://workshop.colips.org/dstc7/) has ended on January 27, 2019. This github project is still available 'as is', but we unfortunately no longer have time to maintain the code or to provide assistance with this project.**

## News
* 10/29/2018: [Spreadsheet](https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling/blob/master/evaluation/dstc7-task2-individual_judgments.xlsx) containing indivdual judgments used for human evaluation.
* 10/23/2018 and 10/15/2018: Automatic and human evaluation [results](https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling/blob/master/evaluation/dstc7-task2-scores.xlsx) posted. The code to reproduce the automatic evaluation and get the same scores can be found [here](https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling/blob/master/evaluation/src/README.md). 
* 10/8/2018: Participants submitted system outputs.
* 9/10/2018-10/8/2018: Evaluation phase, instructions [here](https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling/tree/master/evaluation).
* 7/11/2018: An [FAQ](https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling/tree/master/data_extraction#FAQ) section has been added to the data extraction page. 
* 7/1/2018: [Official training data](https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling/tree/master/data_extraction) is up.
* 6/18/2018: [Trial data](https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling/tree/master/data_extraction/trial) is up.
* 6/1/2018: [Task description](https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling/blob/master/doc/DSTC7_task2.pdf) is up.
* 6/1/2018: [Registration](https://docs.google.com/forms/d/e/1FAIpQLSf4aoCdtLsnFr_AKfp3tnTy4OUCITy5avcEEpUHJ9oZ5ZFvbg/viewform) for DSTC7 is now open.

## Registration

~~Please register [here]~~ Registration has now closed.

## Task
This [DSTC7](http://workshop.colips.org/dstc7/) track presents an end-to-end conversational modeling task, in which the goal is to generate conversational responses that go beyond trivial chitchat by injecting informative responses that are grounded in external knowledge. This task is distinct from what is commonly thought of as goal-oriented, task-oriented, or task-completion dialog in that there is no specific or predefined goal (e.g., booking a flight, or reserving a table at a restaurant). Instead, it targets human-like interactions where the underlying goal is often ill-defined or not known in advance, of the kind seen, for example, in work and other productive environments (e.g.,brainstorming meetings) where people share information.

Please check this [description](https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling/blob/master/doc/DSTC7_task2.pdf) for more details about the task, which follows our previous work ["A Knowledge-Grounded Neural Conversation Model"](https://arxiv.org/abs/1702.01932) and our original task [proposal](https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling/blob/master/doc/proposal.pdf).

## Data
We extend the [knowledge-grounded](https://arxiv.org/abs/1702.01932) setting, with each system input consisting of two parts: 
* Conversational data from Reddit.  
* Contextually-relevant “facts”, taken from the website that started the (Reddit) conversation.

Please check the [data extraction](https://github.com/DSTC-MSR/DSTC7-End-to-End-Conversation-Modeling/tree/master/data_extraction) for the input data pipeline. Note: We are providing scripts to extract the data from a Reddit [dump](http://files.pushshift.io/reddit/comments/), as we are unable to release the data directly ourselves. 

## Evaluation
As described in the [task description](http://workshop.colips.org/dstc7/proposals/DSTC7-MSR_end2end.pdf) (Section 4), We will evaluate response quality using both automatic and human evaluations on two criteria.
* Appropriateness;
* Informativeness.

We will use automatic evaluation metrics such as BLEU and METEOR to have preliminary score for each submission prior to the human evaluation. Participants can also use these metrics for their own evaluations during the development phase. We will allow participants to submit multiple system outputs with one system marked as “primary” for human evaluation. We will provide a BLEU scoring script to help participants decide which system they want to select as primary. 

We will use crowdsourcing for human evaluation. For each response, we ask humans if it is an (1) appropriate and (2) informative response, on a scale from 1 to 5. The system with best average Appropriateness and Informativeness will be determined the winner.

## Baseline
A standard seq2seq [baseline model](https://github.com/DSTC-MSR/DSTC7-End-to-End-Conversation-Modeling/tree/master/baseline) will be provided soon.

## Timeline
|Phase|Dates|
| ------ | -------------- |
|1. Development Phase|June 1 – September 9|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.1 Code (data extraction code, seq2seq baseline)|June 1|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.2 "Trial" data made available|June 18|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.3 Official training data made available| July 1|
|2. Evaluation Phase|September 10 – October 8|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.1 Test data made available|September 10|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.2 Participants submit their system outputs|October 8|
|3. Results are released|October|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.1 Automatic scores (BLEU, etc.)|October 16|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.2 Human evaluation|October 23|

## Organizers
* [Michel Galley](https://www.microsoft.com/en-us/research/people/mgalley/)
* [Chris Brockett](https://www.microsoft.com/en-us/research/people/chrisbkt/)
* [Sean Xiang Gao](https://www.linkedin.com/in/gxiang1228/)
* [Bill Dolan](https://www.microsoft.com/en-us/research/people/billdol/)
* [Jianfeng Gao](https://www.microsoft.com/en-us/research/people/jfgao/)

## Reference
If you submit any system to DSTC7-Task2 or publish any other work making use of the resources provided on this project, we ask you to cite the following task description paper:

```Michel Galley, Chris Brockett, Xiang Gao, Bill Dolan, Jianfeng Gao. End-to-End conversation Modeling: DSTC7 Task 2 Description. In DSTC7 workshop (forthcoming).```

## Contact Information
* ~~For questions specific to Task 2, you can contact us at <dstc7-task2@microsoft.com>.~~ (No longer maintained.)
* ~~You can get the latest updates and participate in discussions on [DSTC mailing list](http://workshop.colips.org/dstc7/contact.html).~~
