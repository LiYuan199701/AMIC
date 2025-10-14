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
<img width="991" height="544" alt="Screenshot 2025-10-14 at 12 16 48 PM" src="https://github.com/user-attachments/assets/5991094b-0de4-463b-9cc1-ec3b6bd2308e" />
