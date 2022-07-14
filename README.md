# Amazon Music Instrument Review

## About dataset

This data file has reviewer ID , User ID, Reviewer Name, Reviewer text, helpful, Summary(obtained from Reviewer text),Overall Rating on a scale 5, Review time

Description of columns in the file:

1. reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
2. asin - ID of the product, e.g. 0000013714
3. reviewerName - name of the reviewer
4. helpful - helpfulness rating of the review, e.g. 2/3
5. reviewText - text of the review
6. overall - rating of the product
7. summary - summary of the review
8. unixReviewTime - time of the review (unix time)
9. reviewTime - time of the review (raw)



Dataset published on [Kaggle](https://www.kaggle.com/datasets/eswarchandt/amazon-music-reviews)

## About Sentiment analysis-

### Problem
This is a sentiment prediction problem based on text reviews of a music instrument products given by the users.
<br>

More detailed explanation in the files - 
[GitHub Notebook](sentiment_analysis_using_classifiers.ipynb) / [Web-nbviewer](https://nbviewer.org/github/bajajvatsal/amazon-music-instrument-review/blob/main/sentiment_analysis_using_classifiers.ipynb)
or 
[Python Script](sentiment_analysis_using_classifiers.py)
