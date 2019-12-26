## Assignment 1
### The very basics with Scikit-Learn

In this tutorial, we will be classifying whther a given 6-letter word is English or German. First, let's get some training and target data. See this table:


<sub> Table 1: Training words and target labels in two languages </sub>

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

<sub> Code block 1: Changing letters into numbers using ```ord()``` </sub>

```python
string_to_ord = []
for char in "ANYONE":
  string_to_ord.append(ord(char))
print(string_to_ord)
```
Print output:
```[65, 78, 89, 79, 78, 69]```

You can think of the above code from letter to ```ord()``` number representation like this:

<sub> Table 2: Letter to ```ord()``` visualization </sub>

|65|78|89|79|78|69|
|---|---|---|---|---|---|
|A|N|Y|O|N|E|

Now, when we make a two-dimensional array with the 6 words found in Table 1, it would look something like this:

<sub> Code block 2: Two-dimensional array representation of the 6 words found in Table 1 </sub>

```python
training = [
  [65, 78, 89, 79, 78, 69],   # ANYONE
  [85, 80, 82, 79, 65, 82],   # UPROAR
  [89, 69, 76, 76, 79, 87],   # YELLOW
  [66, 196, 82, 71, 69, 84],  # BÄRGET
  [90, 85, 82, 85, 70, 69],   # ZURUFE
  [87, 220, 83, 84, 69, 77],  # WÜSTEM
]
```

  
