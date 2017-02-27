# Twitter Hashtag Classifier

## Todo List
    1. Brief introduction to supervised learning
    2. Implement a simple Naive Bayes classifier for twitter hashtags
    3. Math behind Naive Bayes

## Introduction
    Supervised learning => Function approximation
    Given: {x1 = 0, y1 = 1}, {x2 = 1, y2 = 2}, {x3 = 2, y3 = 4} | Problem: {x4 = 3, y4 = ?}
    y_i = f(x_i) = 2^(x_i)
    y_i = f(x_i) = 1 + x1 + x2 + ... + x_i
    Instance => Features + Label
    Classifier learns from example instances (Build f())
    And predicts the label of new instances by given features (Label = f(Features))

## Example
    Given: {"I love painting", #art}, {"President Trump said something awful", #Trump}
    Problem: {"Who is the president?", #art or #Trump ?}
    What can be possible features?

## Installation
    1. sudo pip install pymongo
    2. sudo pip install sklearn
    3. (make sure you are using GT Wifi)

## Get Started
    1. Open a new Python file
    2. Follow live demo

## Math
    Basic: P(y | x) = P(x, y) / P(x) = P(x | y) * P(y) / P(x)
    Naive Bayes: P(Label | Feature_1, Feature_2, ... , Feature_n)
    = P(Feature_1, Feature_2, ... , Feature_n, Label) / P(Feature_1, Feature_2, ... , Feature_n)
    = P(Label) * P(Feature_1 | Label) * P(Feature_2 | Label) * ... * P(Feature_n | Label) * Constant
