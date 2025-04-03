# ELE489 - Homework 1  Alperen Özkan
## k-Nearest Neighbors (k-NN) Classifier from Scratch

This project is part of the ELE489 course: *Fundamentals of Machine Learning* at Hacettepe University.  
The objective was to implement the **k-NN algorithm from scratch** in Python (without using any machine learning libraries for classification) and test it on the Wine dataset from the UCI Machine Learning Repository.

---

## 📌 Overview

In this project:

- The k-NN classifier was written manually using NumPy.
- Three distance metrics were implemented: **Euclidean**, **Manhattan**, and **Chebyshev**.
- The algorithm was tested with various **K** values (1, 3, 5, 7, 9, 11, 13).
- Accuracy scores were calculated and compared for each setting.
- Data visualization techniques such as **KDE plots**, **pair plots**, and a **correlation heatmap** were used to analyze feature distribution and class separation.
- Evaluation was performed using **confusion matrices** and **classification reports**.

---

## 📁 Files Included

- `knn.py` – Custom implementation of the k-NN classifier  
- `analysis.ipynb` – Jupyter Notebook containing all steps: preprocessing, training, evaluation, and visualizations  
- `README.md` – Project documentation (this file)  

> ⚠️ Note: All results and figures are generated directly in the notebook.  
> There are no additional folders (e.g., saved plots), as all visual outputs are displayed inline.

---

## 📂 Dataset Information

The [Wine dataset](https://archive.ics.uci.edu/ml/datasets/wine) includes:

- **178 samples**
- **13 numerical features**
- **3 class labels** (1, 2, 3)

The features include chemical properties of wines (e.g., alcohol, flavanoids, color intensity, magnesium, etc.).

---

## 🔧 How to Run

### Requirements

You can run this project using any Python environment (e.g., Anaconda, Google Colab, VS Code).

Required Python packages:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

To install manually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
