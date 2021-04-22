## Time and Space Complexity Analysis

Here, assume: 
+ t -> No. of iterations
+ N -> No. of Samples
+ M -> No. of Features
+ k -> No. of Classes

**Time Complexity Analysis of Logistic Regression**

1) *Training*:

In one learning iteration, we calculate the predicted labels by multiplying the input matrix with the coefficient matrix and taking the sigmoid of the resulting value. Hence, the multiplication requires *O(NMk)* time complexity. For gradient calculation, we compute Indicator vectors for each class, and hence it also requires *O(NMk)*.

	Time Complexity is O(NMkt)

2) *Predicting*:	

Here, we are computing the predicted labels, which is just the first part of Training.

	Time Compexity is O(NMk)
	

**Space Complexity Analysis of Logistic Regression**

1) *Training*:

The input matrix requires *O(NM)* space, the weights require *O(Mk)* space, and finally the predicted values require *O(Nk)* space. 

    Space Complexity O(NM + Nk + Mk) = O(NM + k(N+M))

2) *Predicting*:	

Here, we require the same space as in training.

	Space Complexity O(NM + Nk + Mk) = O(NM + k(N + M))
	For one sample, space complexity O(k + kM)

