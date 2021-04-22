## Time and Space Complexity Analysis

Here, assume: 
+ t -> No. of iterations
+ N -> No. of Samples
+ M -> No. of Features

**Time Complexity Analysis of Logistic Regression**

1) *Training*:

In one learning iteration, we calculate the predicted labels by multiplying the input matrix with the coefficient matrix and taking the sigmoid of the resulting value. Hence, the multiplication requires *O(NM)* time complexity. For gradient calculation, we compute *X^T(y_hat - y)* which requires the same time complexity.

	Time Complexity is O(NMt)

2) *Predicting*:	

Here, we are computing the predicted labels.

	Time Compexity is O(NM)
	

**Space Complexity Analysis of Logistic Regression**

1) *Training*:

The input matrix requires *O(NM)* space, the weights require *O(M)* space, and finally the predicted values require *O(N)* space. 

    Space Complexity O(NM + N + M) = O(NM)

2) *Predicting*:	

Here, we require the same space as in training.

	Space Complexity O(NM)
	For one sample, space complexity O(M)

