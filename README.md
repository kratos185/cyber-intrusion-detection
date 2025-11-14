Cyber Intrusion Detection System

This repository contains a Streamlit-based web application developed for experimenting with machine learning methods for intrusion detection. The application provides an interface where users can upload datasets, trained models, and encoders, and then evaluate model performance or generate predictions on network traffic samples.

Overview

The purpose of this project is to make intrusion detection workflows easier to test and visualize. Instead of running separate scripts for preprocessing, prediction, and evaluation, everything is accessible through a simple web interface. Users can load their own models (saved as .pkl files), upload CSV datasets, compare model outputs, and explore different metrics and feature distributions directly from the browser.

Features

Multi-page Streamlit design for better navigation

Upload support for models, encoders, and datasets

Evaluation dashboard with performance metrics and plots

Single-sample as well as batch prediction options

Basic exploratory visualizations for network traffic features

Compatible with any Pickle-saved scikit-learn model

Technologies Used

Python

Streamlit

scikit-learn

pandas, numpy

matplotlib, seaborn, plotly (for visualizations)
