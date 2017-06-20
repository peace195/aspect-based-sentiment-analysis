# SentimentAnalysis

## Descriptions
SemEval-2015 Task 12: Aspect Based Sentiment Analysis: http://alt.qcri.org/semeval2015/task12/
I specialize in restaurants domain. You can see final results of contest in [1].
The purpose of this project are:
* A sample of bidirectional LSTM (tensorflow 1.2.0).
* A sample of picking some special units of a recurrent network (not all units) to train and predict their labels. 
* Compare between struct programming and object-oriented programming in Deep Learning model.

![alt text](https://github.com/peace195/SentimentAnalysis/blob/master/model.png)

Step by step:
1. Used contest data and "addition restaurantsreview data" to learn word embedding by fastText.
2. Used bidirectional LSTM in the model as above. Input of the model are the vector of word embedding that we trained before.

## Getting Started

### Data
* SemEval-2015 Task 12 dataset: http://alt.qcri.org/semeval2015/task12/index.php?id=data-and-tools
* Addition restaurant review data for restaurant-word2vec. Here, I use:
	* https://inclass.kaggle.com/c/restaurant-reviews
	* https://www.kaggle.com/snap/amazon-fine-food-reviews

### Prerequisites
* python 2.7
* tensorflow 1.2.0: https://www.tensorflow.org/versions/r0.12/get_started/os_setup#download-and-setup
* fastText: https://github.com/facebookresearch/fastText

### Installing
	
	python sa_aspect_term_oop.py
	
## Results

## Authors

* **Binh Thanh Do** 

## References
* [1] http://alt.qcri.org/semeval2015/cdrom/pdf/SemEval082.pdf

## License

This project is licensed under the MIT License