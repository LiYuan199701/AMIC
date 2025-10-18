# Usage
**This main branch is used to track and update my progress of manual scripts on Overleaf for this project. The main LaTeX file, paper reference bib file, and all figures are in the AMIC_Uncertainty_Bayesian_Variational folder in the main branch.**

The Bayesian model design and codes are in the sub-branches, such as 
- document-level-BAMIC
- word-level-BAMIC

# This branch is for meeting notes. 

##  Oct. 9th with [Dr. Larson](https://s2.smu.edu/~eclarson/index.html)

Dr. Larson agreed to my second word-level Bayesian AMIC model. Suppose I want to add word-level terms in the loss function. In that case, I need to apply [stop gradient operation](https://www.tensorflow.org/api_docs/python/tf/stop_gradient) to prevent the flow of gradients during the backpropagation phase of training by treating sentiment indication variable $\delta_{ij}$ of the computation graph as a constant, effectively freezing its parameters and preventing them from being updated. 

## Oct. 13th with [Dr. Gupta](https://mehak25.github.io/)

I explained the original AMIC model and my two Bayesian AMIC model designs to Dr. Gupta, and she is willing to help with the code and implementation parts of the model. I shared my GitHub codes and code with her.

## Oct. 13th with [Dr. Luo](https://sites.google.com/view/med-nlp-projectx/home)

Dr. Luo suggested I combine mutual attention with my current self-attention mechanism to incorporate negations and a neighborhood window, thereby finding more connections between each word. Regarding the interpretability, she also suggested I consider the phrase-level sentiment, sentiment-level, and document-level sentiment, and how to aggregate them into one. 

## Oct. 14th with [Dr. Sundararajan](https://sites.google.com/site/raanjuragavendar/home)

I met Dr. Sundararajan about my thesis project. He agreed to work with me for a while to see how it works. He suggested I could apply similar ideas in the Bayesian AMIC model to the time series, which is his primary research area. By employing the Bayesian framework in the time series model, we would like to know the uncertainty of prediction or any other parameters. He then shared with me two papers. One is *Bayesian Perceptron: Towards fully Bayesian Neural Networks*, which is about how to apply Bayesian methods to neural networks, and the other is *Neural Granger Causality*, which is about how to apply nonlinear neural networks to a time series application. 

## Oct. 17th with [Nastaran Ghorbani](https://www.linkedin.com/in/nastaranghorbani/), PhD student in Data Science at SMU

She showed me how she applied the transformer model embedding into [Chenyu Yang](https://scholar.google.com/citations?user=yqz6YIIAAAAJ&hl=en)'s AMIC model. By using the transformer-based word embeddings, she improved the accuracy of model prediction. She also shared with me Chenyu Yang's newly submitted paper about his new AMIC that adds local and global shifters to capture the negation and more contextual information in the model. 
