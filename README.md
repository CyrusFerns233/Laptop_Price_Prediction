![Laptop Price Predictor](laptop.png)

# Laptop Price Prediction Project

## Overview
This repository contains a machine learning project that predicts the price of laptops based on various hardware specifications and brands. This project utilizes a Streamfile app for a user-friendly interface, allowing users to input laptop specifications and get predicted prices in real time.

## Prerequisites
Ensure Python is installed on your system. You can install Python from [here](https://www.python.org/downloads/).

## Installation
To set up this project, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/laptop-price-predictor.git
   cd laptop-price-predictor

2. Install the required Python libraries:
   ```bash
    pip install -r requirements.txt

 3. Usage
  To run the Streamlit app locally, execute:
    ```bash
    streamlit run main.py

This will start the web app, which can be accessed via your web browser at `http://localhost:8501`.

## Features
- **Prediction**: Allows users to input laptop specifications through a friendly UI and get price predictions.
- **Data Analysis**: Provides visual analytics on the dataset, such as price distributions, average prices by brand, and specifications distributions.
- **Interactive Visualizations**: Interactive charts powered by Seaborn and Matplotlib for deeper insights.

## How it Works
The project uses a machine learning model trained on historical data of laptop sales. The model is serialized into a pickle file (`pipe.pkl`) and is loaded into the Streamframework at runtime. Users interact with the model through a Streamlit interface where they can specify details like brand, processor, and memory to get a price prediction.

## Contributions
Contributions are welcome! Please fork the repository and open a pull request with your features or fixes.

## References and Acknowledgements

This project was inspired by a YouTube tutorial on building price prediction models with Python and Streamlit. You can view the original tutorial here: [Laptop Price Prediction Tutorial by Nitish Sir @CampusX](https://www.youtube.com/watch?v=BgpM2IiCH6k&t=5030s).

### Modifications to the Original Tutorial
While the foundational concepts were derived from the tutorial mentioned above, several significant modifications were made to tailor the project to specific requirements:
- **Dataset**: The dataset used in this project differs from the one used in the tutorial. Instead of using the provided sample dataset, a more comprehensive and diverse dataset was sourced to improve the model's accuracy and robustness.
- **Data Cleaning Process**: The data cleaning process was extensively customized to address the specific challenges and nuances of the new dataset. This includes more sophisticated handling of missing values, outliers, and feature engineering to better capture the predictive signals in the data.

These adaptations were crucial for enhancing the model's performance and ensuring that it meets the specific needs of predicting laptop prices based on varied and real-world data.

