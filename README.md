# DSTC7: End-to-End Conversation Modeling

## Moving beyond Chitchat
This [DST7](http://workshop.colips.org/dstc7/) track proposes an end-to-end conversational modeling task, where the goal is to generate conversational responses that go beyond chitchat, by injecting informational responses that are grounded in external knowledge (e.g.,Foursquare, or possibly also Wikipedia, Goodreads, or TripAdvisor). There is no specific or predefined goal (e.g., booking a flight, or reserving a table at a restaurant), so this task does not constitute what is commonly called either goal-oriented, task-oriented, or task-completion dialog, but target human-human dialogs where the underlying goal is often ill-defined or not known in advance, even at work and other productive environments (e.g.,brainstorming meetings).

Please check the [proposal](http://workshop.colips.org/dstc7/proposals/DSTC7-MSR_end2end.pdf) for a full description of the task, which follows our previous work ["A Knowledge-Grounded Neural Conversation Model"](https://arxiv.org/abs/1702.01932)

## Task, Data and Evaluate
We extend the knowledge-grounded setting, with each system input consisting of two parts: 
* Conversational input, from Reddit
* Contextually-relevant “facts”, from WWW

Please check the [data extraction](https://github.com/DSTC-MSR/DSTC7-End-to-End-Conversation-Modeling/tree/master/data_extraction) for the input data pipeline.

We will evaluate response quality using both automatic and human evaluation on two criteria .
* Appropriateness
* Informativeness & Utility

A  [baseline model](https://github.com/DSTC-MSR/DSTC7-End-to-End-Conversation-Modeling/tree/master/baseline) is provided.

## Timeline
|Phase|Dates|
| ------ | -------------- |
|1. Development Phase|June 1 – Sept 9|
|1.1 Code (data extraction code, seq2seq baseline), trial data made available|June 1st|
|1.2 Training data made available|July 1st|
|2. Evaluation Phase|Sept 10 – 24|
|2.1 Test data made available|Sept 10|


## Contact Information
* You can contact us at <dstc7-task2@microsoft.com>
* You can get the latest updates and participate in discussions on [DSTC mailing list](http://workshop.colips.org/dstc7/contact.html)
## Organizers
* [Michel Galley](https://www.microsoft.com/en-us/research/people/mgalley/)
* [Chris Brockett](https://www.microsoft.com/en-us/research/people/chrisbkt/)
* [Sean Gao](https://www.linkedin.com/in/gxiang1228/)
* [Bill Dolan](https://www.microsoft.com/en-us/research/people/billdol/)
* [Jianfeng Gao](https://www.microsoft.com/en-us/research/people/jfgao/)
