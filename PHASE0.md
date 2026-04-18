# Phase 0 — Step 1: Mental Model of Learning

## Q1. What is a neural network, really?

A neural network is a **parameterized mathematical function**.

More specifically, it is a function:

f(x; θ)

- x = input  
- θ = parameters (weights and biases)

It is built as a **composition of multiple functions (layers)**:

f(x) = f_L(f_{L-1}(...f_1(x)))

Each layer performs:
1. A linear transformation: Wx + b  
2. A non-linear transformation (activation function)

So if someone handed me a trained neural network as a black box, I would describe it as:

> A structured function that maps inputs to outputs using learned parameters.

---

## Q2. What does “learning” mean — precisely?

Learning is the process of **optimizing parameters** to reduce error.

### What changes:
- Weights and biases (θ)

### What does NOT change:
- Network architecture
- Activation functions
- Loss function definition

### What decides good vs bad:
- A **loss function** that measures the difference between predicted and actual output

### Process:
1. Make a prediction
2. Compute loss
3. Adjust parameters to reduce loss

So:

> Learning = iterative adjustment of parameters to minimize a loss function over data.

---

## Q3. Why do neural networks need to be deep?

While a single-layer network can approximate many functions, depth provides:

### 1. Hierarchical representation
- Early layers: simple patterns
- Later layers: complex patterns

### 2. Parameter efficiency
- Deep networks can represent functions with **far fewer parameters** than shallow ones

### 3. Compositional structure
- Complex functions are built from simpler ones

So:

> Depth allows efficient representation of complex patterns by composing simpler transformations.

---

## Q4. What is a gradient, intuitively?

Imagine a landscape:

- Your position = current parameters  
- Height = loss  

The gradient is:

> The direction of steepest increase in height (loss)

So to reduce loss:
- Move in the **opposite direction of the gradient**

Why it works:
- It gives the best local direction to change parameters to reduce error

So:

> The gradient is like a compass pointing uphill — you go the opposite way to go downhill.

---

## Q5. “Neural networks work because they’re inspired by the human brain.”

**Disagree.**

### Reason:
- Modern neural networks are not biologically accurate
- Real neurons behave very differently (spiking, timing, complex dynamics)
- Artificial neurons are simplified mathematical operations

### What actually makes them work:
- Function approximation capability
- Optimization using gradient descent
- Availability of large datasets and compute power

So:

> Neural networks work because they are powerful trainable functions, not because they mimic the brain.