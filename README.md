# Laptop Price Prediction Project

## Overview
This repository contains a machine learning project that predicts the price of laptops based on various hardware specifications and brands. This project utilizes a Streamfile app for a user-friendly interface, allowing users to input laptop specifications and get predicted prices in real time.

## Installation
To get this project up and running, follow these steps:
1. Clone the repository:

2. Install the required packages:


## Usage
To run the Streamlit app locally, execute:

This will start the web app, which can be accessed via your web browser at `http://localhost:8501`.

## Features
- **Prediction**: Allows users to input laptop specifications through a friendly UI and get price predictions.
- **Data Analysis**: Provides visual analytics on the dataset, such as price distributions, average prices by brand, and specifications distributions.
- **Interactive Visualizations**: Interactive charts powered by Seaborn and Matplotlib for deeper insights.

## How it Works
The project uses a machine learning model trained on historical data of laptop sales. The model is serialized into a pickle file (`pipe.pkl`) and is loaded into the Streamframework at runtime. Users interact with the model through a Streamlit interface where they can specify details like brand, processor, and memory to get a price prediction.

## Contributions
Contributions are welcome! Please fork the repository and open a pull request with your features or fixes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
