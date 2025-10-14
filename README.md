**We used four big datasets to experiment with this new Bayesian AMIC. Each dataset has its own Colab notebook to train and show results.**
- Wine Review Dataset: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://drive.google.com/file/d/1k3KSQ5dcG12yMKW6MVCOGYtQTZOUyerz/view?usp=drive_link](https://colab.research.google.com/drive/1k3KSQ5dcG12yMKW6MVCOGYtQTZOUyerz))
- Amazon Review Dataset
- IMDB Movie Review Dataset
- Twitter Dataset


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
