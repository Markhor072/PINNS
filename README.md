# Physics-Informed Neural Networks (PINNs) Framework

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)
![SciML](https://img.shields.io/badge/SciML-8A2BE2?style=for-the-badge)

A modular and extensible framework for implementing **Physics-Informed Neural Networks (PINNs)** to solve forward and inverse problems involving ordinary and partial differential equations (ODEs/PDEs). This repository serves as a foundational codebase for exploring Scientific Machine Learning (SciML).

---

## 📖 Introduction

Physics-Informed Neural Networks (PINNs) integrate the governing physical laws of a system into the learning process of a deep neural network. By using automatic differentiation to calculate derivatives, the network is trained to satisfy the differential equations, initial conditions, and boundary conditions simultaneously. This approach is powerful for solving problems where data is sparse but the physical model is known.

This framework provides a structured implementation to easily experiment with different PDEs, boundary conditions, and network architectures.

## 🚀 Key Features

- **Modular Design**: Separate, reusable components for data generation, network building, loss computation, and training.
- **Multiple PDE Examples**: Includes implementations for classic equations like:
  - The Allen-Cahn Equation
  - The Burgers' Equation
  - The Heat Equation
  - The Wave Equation
- **Flexible Configuration**: Easily configurable parameters, boundary conditions, and network hyperparameters.
- **Comprehensive Visualization**: Tools for plotting results, comparing solutions, and analyzing loss convergence and PDE residual.
- **Inverse Problem Support**: Framework can be extended to solve inverse problems (parameter discovery).

## 🛠️ Tech Stack & Libraries

- **Core Framework**: TensorFlow 2.x / Keras (for automatic differentiation)
- **Scientific Computing**: NumPy, SciPy
- **Visualization**: Matplotlib, Seaborn
- **Interactive Development**: Jupyter Notebook

## 📁 Repository Structure
PINNS/
├── notebooks/ # Jupyter notebooks for each PDE example

│ ├── 01_Heat_Equation.ipynb

│ ├── 02_Burgers_Equation.ipynb

│ ├── 03_Allen_Cahn_Equation.ipynb

│ └── 04_Wave_Equation.ipynb

├── src/ # Core Python source modules

│ ├── pinns.py # Main PINNs model class

│ ├── data_generator.py # Utilities for generating training data (collocation points, BCs, ICs)

│ ├── losses.py # Custom loss functions (PDE loss, BC loss, IC loss)

│ └── visualize.py # Functions for plotting results

├── outputs/ # Directory for saved models, plots, and results

├── requirements.txt # Python dependencies

└── README.md


## 💡 Core Methodology

The general PINNs methodology implemented here involves:

1.  **Define the PDE**: Specify the governing equation, e.g., for a general PDE:
    $$\mathcal{N}[u](t, x) = 0, \quad x \in \Omega, t \in [0, T]$$
    where $\mathcal{N}$ is a nonlinear differential operator.

2.  **Construct the Network**: A neural network $u_{\theta}(t, x)$ approximates the solution, parameterized by weights $\theta$.

3.  **Formulate the Loss**: The total loss is a weighted sum of:
    - **PDE Loss** ($L_{f}$): Mean squared error of the PDE residual $\mathcal{N}[u_{\theta}]$ at collocation points.
    - **Boundary Condition Loss** ($L_{bc}$): MSE on the boundary $\partial\Omega$.
    - **Initial Condition Loss** ($L_{ic}$): MSE at the initial time $t=0$.

4.  **Train the Model**: Minimize the composite loss $L = L_{f} + L_{bc} + L_{ic}$ using gradient-based optimization (e.g., Adam).

## 🔧 Installation & Getting Started

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Markhor072/PINNS.git
    cd PINNS
    ```

2.  **Install dependencies**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    # Core dependencies can also be installed manually:
    # pip install tensorflow numpy matplotlib jupyter scipy
    ```

3.  **Run an example**
    Launch Jupyter Notebook and explore the provided examples.
    ```bash
    jupyter notebook
    ```
    Open and run `notebooks/01_Heat_Equation.ipynb` to see a basic example.

## 📈 Results

The framework successfully solves various benchmark problems. The `outputs/` directory contains examples of:
- **Solution Plots**: Comparing the PINN solution to analytical or reference numerical solutions.
- **Loss Convergence**: Tracking the evolution of the total loss and its individual components during training.
- **Residual Maps**: Visualizing the absolute error or PDE residual across the domain.

## 👨‍💻 Author

**Shahid Hassan**

- GitHub: [@Markhor072](https://github.com/Markhor072)
- LinkedIn: [Shahid Hassan](https://www.linkedin.com/in/markhor072)
- Portfolio: [shahidhassan.vercel.app](https://shahidhassan.vercel.app)

## 📚 References & Further Reading

1.  **Original PINNs Paper:** Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686–707.
2.  **Scientific Machine Learning:** https://www.sciml.org/
3.  **TensorFlow Tutorials on Automatic Differentiation:** https://www.tensorflow.org/guide/autodiff

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Markhor072/PINNS/issues).

If you find this framework useful, please consider giving it a ⭐ on GitHub!
