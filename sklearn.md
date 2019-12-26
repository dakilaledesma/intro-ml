## Assignment 1
### The very basics with Scikit-Learn

In this tutorial, we will be classifying whther a given 6-letter word is English or German. First, let's get some training and target data. See this table:

| Training (often denoted as 'X') | Target (often denoted as 'y' |
| ---- | ---- |
|ANYONE|English|
|UPROAR|English|
|YELLOW|English|
|BÄRGET|German|
|ZURUFE|German|
|WÜSTEM|German|

This is a simple way to think about your training and target arrays. For classification, your data is structured in a way that X is the data you want to train on, and y is the label (or answer) associated with the data.

However, you'll have to change these words into a number representation in order to use them as data for our models. The ```ord()``` function allows us to convert a character into an integer. Using ```ord()``` and a for loop, we can iterature through a word and change every character into a number representation. We store these integers into an array that represents the original word.

