# Inverse Transform Method for Probability Distribution Generation

This document explains how to generate random variables following specific probability distributions using the inverse transform method.

## The Inverse Transform Method

The inverse transform method is a technique used to generate random samples from a probability distribution by using its cumulative distribution function (CDF).

### Steps:
1. Generate a uniform random number U on [0,1]
2. Compute X = F^(-1)(U), where F^(-1) is the inverse of the CDF
3. X will follow the desired probability distribution

## Triangle Distribution

A triangle distribution is defined by three parameters: a (minimum), b (maximum), and c (mode).

### PDF:
```
f(x) = {
    2(x-a)/((b-a)(c-a))  if a ≤ x ≤ c
    2(b-x)/((b-a)(b-c))  if c < x ≤ b
    0                    otherwise
}
```

### CDF:
```
F(x) = {
    (x-a)²/((b-a)(c-a))      if a ≤ x ≤ c
    1 - (b-x)²/((b-a)(b-c))  if c < x ≤ b
    0                        if x < a
    1                        if x > b
}
```

### Inverse CDF for random generation:
```
F^(-1)(u) = {
    a + √(u·(b-a)(c-a))             if 0 ≤ u ≤ (c-a)/(b-a)
    b - √((1-u)(b-a)(b-c))          if (c-a)/(b-a) < u ≤ 1
}
```

## Pareto Distribution

The Pareto distribution is characterized by two parameters: α (shape) and xm (scale or minimum value).

### PDF:
```
f(x) = {
    (α·xm^α)/x^(α+1)  if x ≥ xm
    0                 if x < xm
}
```

### CDF:
```
F(x) = {
    1 - (xm/x)^α  if x ≥ xm
    0             if x < xm
}
```

### Inverse CDF for random generation:
```
F^(-1)(u) = xm / (1-u)^(1/α)  for 0 ≤ u < 1
```
## Rejection Sampling

Rejection sampling is an alternative technique for generating random samples from probability distributions when the inverse transform method is difficult to apply.

### Steps:
1. Choose a proposal distribution g(x) that is easy to sample from
2. Find a constant M such that f(x) ≤ M·g(x) for all x
3. Generate Y from the proposal distribution g(x)
4. Generate U from Uniform(0,1)
5. If U ≤ f(Y)/(M·g(Y)), accept Y; otherwise, reject and go back to step 3

## Gamma Distribution Example

For a Gamma distribution with parameters λ = 1.5 and α = 2, we can use rejection sampling with an exponential proposal distribution.

### PDF of Gamma(λ,α):
```
f(x) = (λ^α / Γ(α)) · x^(α-1) · e^(-λx)  for x > 0
```

Where Γ(α) is the gamma function.

### Why Exponential is Suitable:
An exponential distribution works well as a proposal distribution because it has a similar shape to the gamma distribution. A Gaussian distribution would not be suitable because it decays too quickly near the edges compared to the gamma distribution's tail behavior.

### Implementation:
For our Gamma(λ=1.5, α=2) distribution, we can use an exponential proposal with rate parameter λ' = 0.75 and a scaling constant M = 4/e ≈ 1.47.

```
g(x) = λ' · e^(-λ'x)  for x > 0
```

The acceptance criterion becomes:
```
U ≤ f(Y)/(M·g(Y)) = (λ^α · Y^(α-1) · e^(-λY)) / (M · λ' · e^(-λ'Y) · Γ(α))
```
In our case, an acceptance rate of around 68% was found, indicating decent fit of the proposal function

These values (λ' = 0.75 and M = 4/e) were determined through experimentation and research to optimize the acceptance rate while ensuring the proposal distribution properly envelops the target gamma distribution.