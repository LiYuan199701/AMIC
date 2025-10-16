# Usage
**The main branch is used to track and update my progress of manual scripts on Overleaf for this whole project. The main LaTeX file, paper reference bib file, and all figures are in the AMIC_Uncertainty_Bayesian_Variational folder and each other folders.**

The Bayesian model design and codes are in the sub-branches, such as 
- document-level-BAMIC
- word-level-BAMIC
- Time-impact-Wine-Review

# AMIC
AMIC is a natural language shallow neural network model that integrates layers of self-attention with
logistic linear regression methods to calculate sentiment scores at the word and document level. This AMIC
model performs well on a wine review dataset with approximately 89% accuracy in the test set. However, the
current architecture of the AMIC model can only generate point estimations in either continuous regression
or classification tasks. As a data scientist and statistician, I always want to know the level of uncertainty
about the estimation of points. This is why we would like to modify the AMIC to produce uncertainty for
prediction point estimations.

# Downstream Analysis of AMIC: Time Impact on Wine Reviews

This linear regression analysis looks at how wine review ratings as scores change over time using subgroup linear regression methods, given the sentiment scores output from the AMIC model. The Google notebook contains all the codes and output, and the folder *AMIC_Wine_Time_Impact* including pdf report and tex files. 
