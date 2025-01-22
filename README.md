# BMO-GNN: Bayesian Mesh Optimization for Graph Neural Networks to Enhance Engineering Performance Prediction

This repository contains the research code for the paper:

> **BMO-GNN: Bayesian Mesh Optimization for Graph Neural Networks to Enhance Engineering Performance Prediction**  
> Jangseop Park and Namwoo Kang.

**Abstract:**  
High-fidelity engineering simulations (e.g., FEA or FEM) are crucial in 3D CAD-based design but often require significant computational resources, making them challenging for design optimization or real-time prediction. This paper presents **BMO-GNN**, a graph neural network (GNN) surrogate model combined with **Bayesian optimization (BO)** to dynamically determine mesh resolution. By converting CAD models into **polygon meshes** and then into **graphs**, our model learns to predict engineering properties (e.g., mass, rim stiffness, disk stiffness). Through BO, the model adaptively searches for optimal parameters (subdivision and clustering) that maximize accuracy while minimizing training costs. BMO-GNN significantly outperforms naive MCMC in terms of both efficiency and predictive performance, achieving up to R² 0.98.

![Graphical Abstract](./figures/graphical_abstract.jpg "Graphical Abstract Example")

---

## Project Structure

An example directory structure is shown below. Adjust to match your setup:

<pre>
.
├── data
│   ├── graphs.pkl
│   ├── stl
│   │   ├── wheel_0001.stl
│   │   ├── wheel_0002.stl
│   │   └── ...
│   └── ...
├── figures
│   ├── graphical_abstract.jpg
│   └── ...
├── run_model.py           # Main script (Bayesian optimization + GNN training)
├── preprocess.py          # Data preprocessing (subdivide, cluster, etc.)
├── config.py              # Configuration loader
├── models.py              # GNN architectures (Spektral-based)
├── utils.py               # Utility functions (training loops, logging, etc.)
├── plot_utils.py          # Visualization, Grad-CAM, animations, etc.
├── requirements.txt
└── README.md              # This document
</pre>

## Data Download Note

- **Necessary Files**  
  - `data/graphs.pkl` or equivalent graph/mesh data  
  - Label files with mass, rim stiffness, disk stiffness, etc.  
  - Original STL or OBJ 3D CAD files (optional)  

Provide a [Kaggle or Drive link](#) if the dataset is publicly available.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/username/BMO-GNN.git
cd BMO-GNN
```
