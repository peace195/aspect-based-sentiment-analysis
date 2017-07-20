# SentimentAnalysis

## Descriptions
SemEval-2015 Task 12: Aspect Based Sentiment Analysis: http://alt.qcri.org/semeval2015/task12/

SemEval-2016 Task 5: Aspect Based Sentiment Analysis: http://alt.qcri.org/semeval2016/task5/

I specialize in restaurants domain. You can see final results of contest in [1][2].
The purpose of this project are:

* Aspect based sentiment analysis.
* A sample of bidirectional LSTM (tensorflow 1.2.0).
* A sample of picking some special units of a recurrent network (not all units) to train and predict their labels. 
* Compare between struct programming and object-oriented programming in Deep Learning model.
* Build stop words, incremental, decremental, positive & negative dictionary for sentiment problem.

![alt text](https://github.com/peace195/SentimentAnalysis/blob/master/model.png)

Step by step:
1. Used contest data and "addition restaurantsreview data" to learn word embedding by fastText.
2. Used bidirectional LSTM in the model as above. Input of the model are the vector of word embedding that we trained before.

## Results
BINGO!!

* Achieved **81.2%** accuracy. **Better** than 2.5% winner team in the semeval2015 competition [1].

* Achieved **85.8%** accuracy. rank 3/28 in the semeval2016 competition [2].

## Getting Started

### Data
* [SemEval-2015 Task 12 dataset](http://alt.qcri.org/semeval2015/task12/index.php?id=data-and-tools)
* [SemEval-2016 Task 5 dataset](http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools)
* Addition restaurant review data for restaurant-word2vec. Here, I use:
	* https://inclass.kaggle.com/c/restaurant-reviews
	* https://www.kaggle.com/snap/amazon-fine-food-reviews
* My embedding result is available here: [google drive](https://drive.google.com/file/d/0B7O__AeIXgEkR3NrU1NEV2JPcXM/view?usp=sharing)

### Prerequisites
* python 2.7
* tensorflow 1.2.0: https://www.tensorflow.org/versions/r0.12/get_started/os_setup#download-and-setup
* fastText: https://github.com/facebookresearch/fastText

### Installing
	
	$ python sa_aspect_term_oop.py
	

## Authors

**Binh Thanh Do** 

## References
[1] http://alt.qcri.org/semeval2015/cdrom/pdf/SemEval082.pdf

[2] http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools

## License

This project is licensed under the MIT License