**The model code for this document-level Bayesian AMIC is in the Colab Notebook** *Model_only_document_level_BAMIC.ipynb*.

**The full training code, including data importing, data preprocessing, data training, and evaluation, is in the following notebooks for each data.**
**We used four big datasets to experiment with this new Bayesian AMIC. Each dataset has its own Colab notebook to train and show results.**
- Wine Review Dataset:       [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1k3KSQ5dcG12yMKW6MVCOGYtQTZOUyerz)
- Amazon Review Dataset:     [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yN_7I0kbxV6CVAClU5ERPXexxetah4CK)
- IMDB Movie Review Dataset: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jdtDmbAeV7gy8BZqK8lqsXSXdyJwixHC)
- Twitter Dataset:           [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zAsVTlK8z6iyH2JK_RCjTegHSzsz80li)


# B-AMIC Model: Integrate Bayesian Statistics with Self-Attention Neural Network on natural languages

## Introduction
AMIC is a natural language shallow neural network model that integrates layers of self-attention with
logistic linear regression methods to calculate sentiment scores at the word and document level. This AMIC
model performs well on a wine review dataset with approximately 89% accuracy in the test set. However, the
current architecture of the AMIC model can only generate point estimations in either continuous regression
or classification tasks. As a data scientist and statistician, I always want to know the level of uncertainty
about the estimation of points. This is why we would like to modify the AMIC to produce uncertainty for
prediction point estimations.

## Introduce original AMIC model
The AMIC model has two layers of nested regressions inside. The out-of-context word embeddings $x_{ij}$ from the outer source are first trained to have contextual word embeddings $a_{ij}^P$ with the transformer-styled self-attention layer. The first layer in the blue box is to select the sentiment word denoted by $\delta_{ij}$ in the document with the logistic regression, given the input contextual word embeddings $a_{ij}^P$. The math formula for this layer is 
The equations are:

<div align="center">

$u_{ij} = a_{ij}^P b$,

$\delta_{ij} = \frac{1}{1+e^{-u_{ij}}}$

</div>

where the $b$ is the logistic coefficients.

The second layer in the red box is to produce the sentiment score $Z_{ij}$ for each word and get the document-level sentiment $Z_i$ with another logistic regression. The math formula for this layer is:

<div align="center">

$Z_{ij} = \delta_{ij} a_{ij}^L \beta$,

$Z_i = \frac{\sum_{j=1}^m Z_{ij}}{m}$,

$\hat{y}_i=I\left(\text{sig}\left(Z_i\right)\right)$

</div>
 
where $\beta$ is the layer two logistic regression coefficient, $a_{ij}^L$ is the contextual word embeddings trained with another self-attention layer, $m$ is the total number of words in a document, $sig$ is the sigmoid function, $\hat{y_i}$ is the predicted the document sentiment label. This model also has three penalty terms: $p_{i1}, p_{i2}, p_{i3}$. $p_{i1}$ encourages peak-ness sentiment word selection. $p_{i2}$ encourages the sparsity of sentiment words. $p_{is}$ stabilizes the sentiment scores $Z_{ij}$. 

<img width="991" height="544" alt="Screenshot 2025-10-14 at 12 16 48 PM" src="https://github.com/user-attachments/assets/5991094b-0de4-463b-9cc1-ec3b6bd2308e" />

Now I introduce the basic implementation part in the AMIC model. We used the PyTorch platform to write the code. We have four essential classes, *SelfAttention*, *Mask_block*, *Sentiment_block*, *Synthesizer*. The blue box layer one is implemented in the *Mask_block*, and the logistic regression part of layer two is implemented in *Sentiment_block*. The final document-level prediction is implemented into the *Synthesizer*. Both *Mask_block* and *Sentiment_block* contain a *SelfAttention* layer to train the contextual self-attention word-level embeddings, $a_{ij}^P, a_{ij}^L$.

## Interface between AMIC and new Bayesian Variational Layer
In this section, we discuss how the AMIC model transitions into the new Bayesian variational layer. The interface happens at layer two. Layer two has one primary output, $a_{ij}^L$, which is used in the Bayesian layer to represent the contextual word-level embeddings. The other output that is used in the Bayesian layer is the padding mask vector from the *Mask_block*. Next, we used the word-level embeddings $a_{ij}^L$ and padding mask $p_i$ to compute the pooled document-level embeddings $h_i$ for the Bayesian layer. The following formula outlines the calculation of this pooled document-level embedding in three steps. The first step is to use the padding mask vector $p_{i}$ to zero out all padding tokens: $$p_{ij} \times a_{ij}^L$$ The second step is to sum this result over the token axis $j$: $$\sum_{j=1}^T (p_{ij} \times a_{ij}^L)$$ The third step is to divide this by number of valid tokens: $$h_i=\frac{\sum_{j=1}^T (p_{ij} \times a_{ij}^L)}{Max(1,\sum_j p_{ij})}$$ where $i$ means the index of document and $j$ means the index of word in the document $i$. However, there are several different methods for computing the pooled document-level embeddings. This is one of them, and we can keep trying other ways. We pass this document-level embedding $h_i$ into a dropout layer, then into our new Bayesian Variational Layer to calculate the uncertainty and the final loss function. 

<img width="993" height="409" alt="Screenshot 2025-10-15 at 10 13 26 PM" src="https://github.com/user-attachments/assets/7a5ffb38-53f8-4eb2-b981-589ea12ae5ef" />

I drew a flow diagram that is similar to Figure 2.1 AMIC to display the high-level model structure of this Bayesian Variational AMIC model in the above figure. The first blue box layer and the second red box layer are identical to the original AMIC model. The new Bayesian layer is the third green box layer. The contextual word embeddings $a_{ij}^L$ are the input to Layer 3. Then, all $a_{ij}^L$ values from one document are pooled to produce the document-level embedding $h_i$. $h_i$ is sent into the Bayesian Variational layer to produce the final prediction $\hat{y}_i$ of document i that is used to compute the final loss function. During the training, posterior parameters $\mu$ and $\rho$ are trained to give weights of this Bayesian layer given the prior input. This is a simplified illustration of Bayesian AMIC; the more detailed low-level diagram is shown in the later Figure.
