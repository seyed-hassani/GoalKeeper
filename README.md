# 🧤 Goalkeeper Neural Network – Regularization in Deep Learning

This project demonstrates how to prevent overfitting in deep neural networks using **dropout** and **L1/L2 regularization**. The goal is to train a model that helps the French national team's goalkeeper accurately throw the ball to a teammate in midfield.

## 🏟️ Problem Description

Given 2D coordinates of past throw events (from 10 previous games), the model learns to predict whether the ball will reach a teammate (`label=1`) or an opponent (`label=0`). The data is synthetic but structured similarly to real-world sports analytics.

## 🧠 Models & Techniques

- A large feedforward network with 3 hidden layers of 5000 neurons each
- Three variations:
  1. **Overfitting-prone base model** with no regularization
  2. **Dropout model** with 60% dropout after each hidden layer
  3. **L1/L2 regularized model** with weights:
     - L1: `3e-5`
     - L2: `3e-4`

## 🛠️ How to Run

1. Install dependencies:
   ```bash
   pip install keras tensorflow numpy matplotlib
