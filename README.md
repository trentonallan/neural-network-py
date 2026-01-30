# Neural Network from Scratch

2-layer feedforward network in pure NumPy with hand-coded backpropagation. No frameworks, no autograd - all 12 partial derivatives manually derived and implemented.

## Implementation

**Architecture:**
- Input: 2 features
- Hidden: 2 neurons (sigmoid)
- Output: 1 neuron (sigmoid)
- Parameters: 6 weights + 3 biases

**Training:**
- SGD with learning rate 0.2
- MSE loss
- 3000 epochs on 4 samples
- Convergence: 0.386 → 0.003 loss

## Gradient Computation

Manually derived all gradients using chain rule for backpropagation:

**Output layer:**
```python
d_L_d_ypred = -2 * (y_true - y_pred)
d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
```

**Hidden layer (backprop through connections):**
```python
d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
# Chain: ∂L/∂w1 = ∂L/∂ŷ · ∂ŷ/∂h1 · ∂h1/∂w1
self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
```

Each weight update requires chaining derivatives through the computational graph. Getting the chain rule order and signs correct determines whether loss converges.

## Results

```
Epoch 0: loss 0.386
Epoch 1000: loss 0.036
Epoch 2000: loss 0.008
Epoch 3000: loss 0.003

Predictions:
Female sample: 0.982
Male sample: 0.019
```

Smooth convergence validates gradient calculations.

## Key Challenges

**Vanishing gradients:** Sigmoid derivative σ'(x) = σ(x)(1-σ(x)) approaches zero at extremes. Deeper networks would need different activations (ReLU, etc).

**Manual differentiation:** Each parameter needs explicit gradient derivation. Error in any single derivative breaks convergence - no autograd to catch mistakes.

**Architecture constraints:** Fully connected means O(n²) parameters between layers. Had to cache all intermediate sums during forward pass for backward computation.

## What This Demonstrates

- Calculus fundamentals (chain rule, partial derivatives)
- Backpropagation algorithm from first principles
- Gradient descent optimization
- Why frameworks like PyTorch exist (autograd is non-trivial)

Built to understand how `.backward()` actually works under the hood.

## Running

```bash
pip install numpy
python neural_network.py
```

## Author

**Trenton Allan**  
Northeastern University  
[allan.tr@northeastern.edu](mailto:allan.tr@northeastern.edu)

---

**Changes:**
- Cut the explanatory/teaching tone
- Assumes reader knows what backprop is
- Shows code instead of explaining concepts
- Focuses on technical decisions and challenges
- More concise, less "here's what I learned"
- Demonstrates competence through brevity
- Still honest about scope but doesn't dwell on it

Now it reads like documentation from someone who knows what they're doing, not a tutorial.
