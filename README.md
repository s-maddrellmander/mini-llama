# Llama Implementation Guide - Enhancing Original Work

[![Python application](https://github.com/s-maddrellmander/mini-llama/actions/workflows/test.yml/badge.svg)](https://github.com/s-maddrellmander/mini-llama/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/s-maddrellmander/mini-llama/branch/main/graph/badge.svg?token=YZALQ4OB7B)](https://codecov.io/gh/s-maddrellmander/mini-llama)

Based on the blog post: ["Llama from scratch (or how to implement a paper without crying)"](https://blog.briankitano.com/llama-from-scratch/). Huge props to [github.com/bkitano](https://github.com/bkitano) for the orignal implimentation, massive thanks!

This README is designed as a comprehensive guide to implement and enhance a simplified version of Llama, initially developed for the TinyShakespeare dataset. While it takes inspiration from Karpathy's Makemore series, the prime focus is to provide clarity on building upon the original author's work.

---

## Installation
```bash
mkdir ~/.venv                      # Create the folder for the environment
python3 -m venv ~/.venv/env        # Create the environment
source ~/.venv/env/bin/activate    # Activate the environment

# Update pip to the latest version
python3 -m pip install --upgrade pip

# Install the IPU specific and graphium requirements
pip install -r requirements.txt

```

---

## Summary

- Emphasis on iterative development: Begin modestly and escalate progressively.
- Utilize tools and techniques shared by the original author.
- Adapt and enhance the paper's methods through a systematic approach.
- Ensure accuracy and efficiency at every layer, drawing upon the foundational work of the original author.

---

## Implementation and Enhancement Strategy

1. **Starting with the Basics**: 
   - Revisit the original author's helper functions. These include:
     - Data splitting
     - Model training
     - Loss visualization
   
2. **Benchmark Model**:
   - Based on the original author's advice, start with a straightforward model, previously mastered.
   - Use the given function to evaluate this model qualitatively.
   - Note the enhancements and changes compared to the original work.
   
3. **Dismantling & Reconstructing**:
   - Dissect the paper into distinct components as guided by the original author.
   - Implement and incrementally enhance each part, always training and evaluating to gauge improvements over the original method.

4. **Layer Enhancements**:
   - Adopt the original author's approach of consistently using `.shape`.
   - Incorporate `assert` and `plt.imshow` for deeper layer insights.
   - Start computations without matrix multiplication. Once you're confident, streamline with enhanced torch functions, if possible.
   - While considering tests for each layer (like the Transformer's attention map), think of optimization strategies the original work might have missed.
   - Extend layer tests across an expanded set of batch, sequence, and embedding sizes to ensure robustness.

---

## Introduction to Llama

**Llama** stands out as a transformer-based model primarily focusing on language modeling. The brainchild of Meta AI, Llama was crafted with a vision of optimizing inference costs, diverging from conventional models that majorly emphasize training costs. Our objective here is to not only implement but also to enhance this intriguing architecture by building on the original author's foundation.

---

## Commencing Your Journey

1. Dive deep into the TinyShakespeare dataset, understanding the original approach and looking for potential areas of improvement.
2. Set the stage – ensure the integration of PyTorch and other pertinent libraries.
3. Initiate with your benchmark model and assess it using the TinyShakespeare dataset, comparing it with the original results.
4. Gradually incorporate the facets of Llama, all the while juxtaposing with the original methodology and noting enhancements.
5. Train, refine, evaluate, and compare consistently.
6. Upon reaching a satisfactory level, perform a holistic evaluation and weigh your results against the original Llama model.

---
