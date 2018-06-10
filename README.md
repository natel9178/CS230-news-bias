# CS230 Project: DeepNews.AI: Detecting Political Bias

Stanford University

## Abstract

Recent trends in media consumption as well as increasing Internet access to news sources necessitates a reliable and efficient method of detecting biases in the news we consume day-to-day. Without a system in which biases are exposed and controlled, economic incentives will continue to favor news publications and authors that engage in inflammatory and often false journalism. In this paper, we develop and explore two different neural network models that attempt the same classification goal: detecting where each news article lies on the political spectrum from conservative to liberal. The first neural network model takes a convolutional approach and the second is structured with a sequential LSTM (long short-term memory) recurrent neural network (RNN) architecture. For our LSTM RNN's, we design both bidirectional and single directional models. Our results show that CNN's produce the most accurate predictions, and that deeper networks can increase accuracy marginally at the expense of precision and recall.

## Additional Data

See additional data, and project milestones under `Project Report`

## Branch Descriptions

**binary**: contains code modificaitons for binary classifications (bias/notbias)

**final-experiments**: contains code modificaitons for 3-Class classifications (Liberal/Neutral/Conservative)
