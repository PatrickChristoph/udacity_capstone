# Customer Segmentation & Prediction

## Description
The project aims to enhance Bertelsmann's understanding of its customer base through a comprehensive data analysis framework. It is divided into three key components:

**Data Exploration**: This initial phase focuses on identifying and addressing data quality issues within the provided demographic data. By cleaning and refining the dataset, the project ensures that subsequent analyses are based on accurate and reliable information.

**Customer Segmentation with Unsupervised Model**: In this part, an unsupervised machine learning model is employed to categorize individuals into distinct segments based on their behaviors and characteristics, allowing the company to uncover distinct customer traits and behaviors.

**Customer Prediction with Supervised Model**: Finally, a supervised machine learning model is utilized to predict future customers, enabling Bertelsmann to effectively target future mailout campaigns based on demographic data.

Overall, the project seeks to leverage data-driven insights to support more effective marketing and customer relationship strategies.


## Project Structure
- `data/`: Contains datasets used for training and testing the model.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model development.
- `src/`: Source code for data processing, model training, and evaluation.
- `requirements.txt`: List of Python packages required to run the project.
- `README.md`: This file.

## Dependencies
This project uses Poetry for dependency management. The following dependencies are specified in the `pyproject.toml` file:

- **Python**: Use 3.12 or higher
- **Pandas**: Data manipulation and analysis
- **Openpyxl**: Reading and writing Excel files
- **Matplotlib**: Plotting library for creating visualizations
- **Scikit-learn**: Machine learning library for Python
- **TQDM**: Progress bar for loops
- **XGBoost**: Optimized gradient boosting library
- **Optuna**: Hyperparameter optimization framework

## Installation
To get started with this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/PatrickChristoph/udacity_capstone.git
   
2. Navigate to project directory:
   ```bash
   cd udacity_capstone

3. Install Poetry:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -

4. Install project dependencies:
   ```bash
   poetry install
   

## Usage
To run Jupyter Notebooks with the libraries installed via Poetry, follow these steps:
1. First, ensure you have Jupyter installed. You can add it to your Poetry environment by running:
   ```bash
   poetry add jupyter
2. Start Jupyter Notebook within the Poetry environment:
   ```bash
   poetry run jupyter notebook
3. This command will open Jupyter Notebook in your web browser. You can now create new notebooks or open existing ones, and all the libraries specified in your pyproject.toml will be available for use.

Alternatively you can possibly use Jupyter Integrations of your IDE, e.g. PyCharm. Usually, you only have to select the Python Interpreter from your created Poetry environment. You can use this command to see the environment path:
`poetry env list --full-path`


## Author
This project was developed by Patrick Christoph as part of the Udacity Data Science program.