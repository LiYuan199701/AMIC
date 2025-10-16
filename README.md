# Usage
**This main branch is used to track and update my progress of manual scripts on Overleaf for this project. The main LaTeX file, paper reference bib file, and all figures are in the AMIC_Uncertainty_Bayesian_Variational folder.**

The Bayesian model design and codes are in the sub-branches, such as 
- document-level-BAMIC
- word-level-BAMIC

# Slide Presentation

The following is the Google slide for this project:

[![Open the Google slides](https://img.shields.io/badge/Slides-Open%20Deck-blue)]([https://docs.google.com/presentation/d/e/2PACX-1vSOoyqhRBvobfHtLVPkk7gl7WdjDZFPKPe7lWK5p1EDZ7HOqz0i7QyGiJ1swJIVGUlVZXg93Ks8C7l-/present](https://docs.google.com/presentation/d/e/2PACX-1vSOoyqhRBvobfHtLVPkk7gl7WdjDZFPKPe7lWK5p1EDZ7HOqz0i7QyGiJ1swJIVGUlVZXg93Ks8C7l-/pub?start=false&loop=false&delayms=3000))




# AMIC
AMIC is a natural language shallow neural network model that integrates layers of self-attention with
logistic linear regression methods to calculate sentiment scores at the word and document level. This AMIC
model performs well on a wine review dataset with approximately 89% accuracy in the test set. However, the
current architecture of the AMIC model can only generate point estimations in either continuous regression
or classification tasks. As a data scientist and statistician, I always want to know the level of uncertainty
about the estimation of points. This is why we would like to modify the AMIC to produce uncertainty for
prediction point estimations.

 
