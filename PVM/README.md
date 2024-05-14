# What is PVM?

### PVM stands for Probabilistic Vectorized Matching, a novel technique for transforming table rows into vectors for more computationally efficient and accurate probabilistic row matching given 2 or more tables without common columns.

## When should I use it?

### PVM is useful in situations when you have 2 or more data frames without common columns and mixed column orderings but still have a certain degree of statistical correlation between certain regions in the matrix of one table and certain regions in the matrix of another table. Using PVM, the tables can still be joined, as if you would apply separate A left join B on A.column=B.column rules for each individual row of both tables.

## How does it work?

### PVM implements Variational Autoencoders (VAE) for transformation of table rows into vectors. Transformed rows are then matched by vector similarity using a greedy similarity based retrieval algorithm.

## Why does it work?

### An important prerequisite for PVM to work is that the columns for tables that are assumed to have a degree of statistical correlation between certain columns to have similar column ordering. This is because upon vectorization, each vector component represents a dimension of its own, and in order to join two tables in a way such that the original statistical correlations between the columns of the joinee table and the columns of the joiner table to be fully captured, we need to ensure that the rows of both tables are vectorized in the latent space with similar dimensional orientation. To achieve this, we run regressions for individual columns of the joinee table on all of the columns of the joiner table. Then we apply the Hungarian algorithm that minimizes the cost matrix to rearrange the joiner table's columns in a way such that the total correlation between each column i of the joinee table and each column j of the joiner table where i=j is maximized. Then we apply VAE for vectorization of the rows of both the joinee and the joiner table. Finally, we use similarity based retreival and match rows using a greedy approach.

## What can I do with this?

### For demonstration, we have shown how we can improve the results of the RANDHIE experiment, which was done in the past, using a dataset from the present. Normally, without common columns, it is difficult to join the two tables in a way such that we can extract meaningful statistical information from the joined columns. However, we demonstrate that joining additional predictors using PVM can add meaningful predictors to an unrelated dataset.

### We have also implemented a Transformer model made by @RayCarpenterIII for predicting heart attack risk from a joiner dataset, thereby showing how PVM allows synthetic variables predicted by a black box deep learning model can also be used for causal inference in a way that is interpretable in the context of a linear regression.

### Please contact me @ wonjae@unc.edu or +15206454193 if you have interest in contributing or peer reviewing our research.