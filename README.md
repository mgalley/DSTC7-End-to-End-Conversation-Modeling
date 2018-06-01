# DSTC7: End-to-End Conversation Modeling

## News
* 6/1/2018: [Task description](https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling/blob/master/doc/DSTC7_task2.pdf) is up.
* 6/1/2018: Registration for DSTC7 Task 2 is now open. To register, simply email us at <dstc7-task2@microsoft.com>.

## Task
This [DSTC7](http://workshop.colips.org/dstc7/) track presents an end-to-end conversational modeling task, in which the goal is to generate conversational responses that go beyond trivial chitchat by injecting informative responses that are grounded in external knowledge. This task is distinct from what is commonly thought of as goal-oriented, task-oriented, or task-completion dialog in that there is no specific or predefined goal (e.g., booking a flight, or reserving a table at a restaurant). Instead, it targets human-like interactions where the underlying goal is often ill-defined or not known in advance, of the kind seen, for example, in work and other productive environments (e.g.,brainstorming meetings) where people share information.

Please check this [description](https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling/blob/master/doc/DSTC7_task2.pdf) for more details about the task, which follows our previous work ["A Knowledge-Grounded Neural Conversation Model"](https://arxiv.org/abs/1702.01932) and our original task [proposal](https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling/blob/master/doc/proposal.pdf).

## Data
We extend the [knowledge-grounded](https://arxiv.org/abs/1702.01932) setting, with each system input consisting of two parts: 
* Conversational data from Reddit.  
* Contextually-relevant “facts”, taken from the website that started the (Reddit) conversation.

Please check the [data extraction](https://github.com/DSTC-MSR/DSTC7-End-to-End-Conversation-Modeling/tree/master/data_extraction) for the input data pipeline.

## Evaluation
We will evaluate response quality using both automatic and human evaluations on two criteria.
* Appropriateness;
* Informativeness & Utility.

## Baseline
A standard seq2seq [baseline model](https://github.com/DSTC-MSR/DSTC7-End-to-End-Conversation-Modeling/tree/master/baseline) will be provided soon.

## Timeline
|Phase|Dates|
| ------ | -------------- |
|1. Development Phase|June 1 – Sept 9|
|1.1 Code (data extraction code, seq2seq baseline)|June 1|
|1.2 "Trial" training data made available|June 1|
|1.3 Official training data made available| By July 1|
|2. Evaluation Phase|Sept 10 – 24|
|2.1 Test data made available|Sept 10|

Note: We are providing scripts to extract the data from a Reddit [dump](http://files.pushshift.io/reddit/comments/), as we are unable to release the data directly ourselves. 

## Organizers
* [Michel Galley](https://www.microsoft.com/en-us/research/people/mgalley/)
* [Chris Brockett](https://www.microsoft.com/en-us/research/people/chrisbkt/)
* [Sean Xiang Gao](https://www.linkedin.com/in/gxiang1228/)
* [Bill Dolan](https://www.microsoft.com/en-us/research/people/billdol/)
* [Jianfeng Gao](https://www.microsoft.com/en-us/research/people/jfgao/)

## Contact Information
* For questions specific to Task 2, you can contact us at <dstc7-task2@microsoft.com>.
* You can get the latest updates and participate in discussions on [DSTC mailing list](http://workshop.colips.org/dstc7/contact.html).
