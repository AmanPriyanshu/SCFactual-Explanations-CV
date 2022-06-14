# SCFactual-Explanations-CV
Creating a pipeline for generating semi-factual and counter-factual explanations for computer vision tasks. A counterfactual explanation describes a causal situation in the form: "If X had not occurred, Y would not have occurred". Basically generating an image close enough to the original one, which results in a target class.

## AutoEncoder Based Implementation:

Now as we can see the generated image does not belong to the same distribution as the dataset, therefore constructing a latent vector to be optimized through an AE is used.

### Results:


![results-ae](/images/AE_Constructions_README.png)

The results below pose interesting observations, discussing a couple of samples:
* Original=2: Attempting to generate an 8, we can see how the top of 2 begins to curve to form an enclosed loop.
* Original=3, Attempting to generate a 5, we can see how the top-right side of 3 begins to disappear creating the empty space present in 5 between its top line and middle curve.
* Original=5, Attempting to generate a 0, we can see how the the top of 5 has disappeared into itself, and lower curve has begun to bend backwards.
* Original=8, Attempting to generate a 1, we can see that the outer parts of 8 have almost disappeared and becoming straight slowly.
* Original=9, Attempting to generate a 7, we can distinctly see how the inward curve of the top-left 9 has opened up, creating a 7. 

### AutoeEncoder:

![ae](/images/ae.png)

## Basic Implementation:

Using simple gradient descent we optimize to find CounterFactuals

![counterfactual](/images/Constructions.png)
