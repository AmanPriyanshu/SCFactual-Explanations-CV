# SCFactual-Explanations-CV
Creating a pipeline for generating semi-factual and counter-factual explanations for computer vision tasks.

## Basic Implementation:

Using simple gradient descent we optimize to find CounterFactuals

![counterfactual](/images/Constructions.png)

## AutoEncoder Based Implementation:

Now as we can see the generated image does not belong to the same distribution as the dataset, therefore constructing a latent vector to be optimized through an AE is used.

### AutoeEncoder:

![ae](/images/ae.png)

### Results:

![results-ae](/images/AE_Constructions.png)
