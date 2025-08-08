Code for the paper:  
**"Optimizing Segmentation of Neonatal Brain MRIs with Partially Annotated Multi-Label Data"**  
Dariia Kucheruk, Sam Osia, Pouria Mashouri, Elizaveta Rybnikova, Sergey Protserov, Jaryd Hunter, Maksym Muzychenko, Jessie Ting Guo, Michael Brudno 
Machine Learning for Healthcare 2025

## Abstract  
Accurate assessment of the developing brain is important for research and clinical applications, and manual segmentation of brain MRIs is a painstaking and expensive process. We introduce the first method for neonatal brain MRI segmentation that simultaneously leverages fully and partially labeled data within a multi-label segmentation framework. Our method improves accuracy and efficiency by utilizing all available supervision—even when only coarse or incomplete annotations are present—enabling the model to learn both detailed and high-level brain structures from heterogeneous data. We validate our method on scans from the Developing Human Connectome Project (dHCP) acquired at both preterm and term gestational ages. Our approach demonstrates more accurate and robust segmentation compared to standard supervised and semi-supervised models trained with equivalent data. The results showed an improvement in predictions of predominantly unannotated labels in the training set when combined with labels of relevant "super-classes". Further experiments with semi-supervised loss functions demonstrated that limited but reliable supervision is more effective than using noisy labels. Our work presents evidence that it is possible to build robust medical image segmentation models with only a small amount of fully labeled training data.

## Overview  
This repository contains the **training pipeline** and **loss function implementations** described in the paper.  
It is intended as a **research prototype** to support reproducibility and understanding of the proposed method.  

## Usage  
The code is provided **as-is** for research purposes.  
It has been tested only in the experimental setup described in the paper and is **not a ready-to-use production system**.
