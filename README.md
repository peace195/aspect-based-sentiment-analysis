# SentimentAnalysis
## Status
This project is ongoing now!

## Descriptions
[SemEval-2014 Task 4: Aspect Based Sentiment Analysis](http://alt.qcri.org/semeval2014/task4/)

[SemEval-2015 Task 12: Aspect Based Sentiment Analysis](http://alt.qcri.org/semeval2015/task12/)

[SemEval-2016 Task 5: Aspect Based Sentiment Analysis](http://alt.qcri.org/semeval2016/task5/)

I specialize in restaurants and laptops domain. You can see final results of these contests in [1][2].
The purposes of this project are:

* Aspect based sentiment analysis.
* A sample of bidirectional LSTM (tensorflow 1.2.0).
* A sample of picking some special units of a recurrent network (not all units) to train and predict their labels. 
* Compare between struct programming and object-oriented programming in Deep Learning model.
* Build stop words, incremental, decremental, positive & negative dictionary for sentiment problem.

Step by step:
1. Used contest data and "addition restaurants review data" to learn word embedding by fastText.
2. Used bidirectional LSTM in the model as above. The input of the model is the vector of word embedding that we trained before.

![alt text](https://raw.githubusercontent.com/peace195/aspect-based-sentiment-analysis/master/model.png)

## Results
BINGO!!
* Outperforms state-of-the-art in semeval2014 dataset [3].

* Achieved **81.2%** accuracy. **Better** than 2.5% winner team in the semeval2015 competition [1].

* Achieved **85.8%** accuracy. rank 3/28 in the semeval2016 competition [2].


## Getting Started

### Data
* [SemEval-2015 Task 12 dataset](http://alt.qcri.org/semeval2015/task12/index.php?id=data-and-tools)
* [SemEval-2016 Task 5 dataset](http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools)
* Addition restaurant review data for restaurant-word2vec. Here, I use:
	* https://inclass.kaggle.com/c/restaurant-reviews
	* https://www.kaggle.com/snap/amazon-fine-food-reviews
* My embedding result is available here: [google drive](https://drive.google.com/drive/folders/0B7O__AeIXgEkRjdLenQ5Ynl4aFk?usp=sharing)

### Prerequisites
* python 2.7
* [tensorflow 1.2.0](https://www.tensorflow.org/versions/r1.2/install/install_linux)
* [fastText](https://github.com/facebookresearch/fastText)

### Installing
	
	$ python sa_aspect_term_oop.py
	

## Authors

**Binh Thanh Do** 

## References
[1] http://alt.qcri.org/semeval2015/cdrom/pdf/SemEval082.pdf

[2] http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools

[3] http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools

## License

This project is licensed under the GNU License