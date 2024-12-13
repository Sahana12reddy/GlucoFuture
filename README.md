
---

# 

GlucoFuture is a user-friendly platform for predicting diabetes risk using advanced machine learning models. It provides an interactive dashboard that enables users to input health metrics and receive instant predictions along with insights into key risk factors.

## Table of Contents

- [Project Overview](#project-overview)
- [Screenshots](#screenshots)
- [Setup](#setup)
- [Usage](#usage)
- [Features](#features)

## Project Overview

GlucoFuture is designed to make diabetes risk assessment simple and accessible. Built using Django for authentication and backend management, it integrates a Streamlit-powered dashboard for real-time predictions and interactive data visualizations.

## Screenshots

### Home Page
![Home](https://github.com/yourusername/GlucoFuture/blob/main/screenshots/home_page.jpeg)

### Login/Registration Page
![Login/Registration](https://github.com/yourusername/GlucoFuture/blob/main/screenshots/login_page.jpeg)

### Dashboard
![Dashboard](https://github.com/Sahana12reddy/GlucoFuture/blob/main/picture/p1.jpeg)
### Prediction Page
![Prediction](https://github.com/yourusername/GlucoFuture/blob/main/screenshots/prediction_page.jpeg)

## Setup

Follow these steps to set up GlucoFuture on your local machine.

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/GlucoFuture.git
   cd GlucoFuture
   ```

2. **Install Requirements**
   Make sure you have Python installed. Then, install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up the Database**
   Initialize and apply migrations for the Django backend:
   ```bash
   python manage.py migrate
   ```

4. **Run the Streamlit Dashboard**
   Open a new terminal window, navigate to the project directory, and run the dashboard file:
   ```bash
   streamlit run dashboard.py
   ```

5. **Run the Django Development Server**
   Start the Django server:
   ```bash
   python manage.py runserver
   ```

   Access the application through [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Usage

Once logged in, users can access the **GlucoFuture** dashboard to:

- Input health metrics like glucose levels, BMI, blood pressure, and age.
- View instant predictions about diabetes risk using an ML model.
- Analyze visual insights into the importance of each health metric in the prediction.

The dashboard ensures a smooth and interactive experience, guiding users through each step of the process.

## Features

- **Diabetes Risk Prediction**: Input personal health data to receive predictions using a trained machine learning model.
- **Data Visualization**: Explore visualizations of feature importance and trends in health data.
- **Interactive Interface**: A clean and intuitive dashboard for seamless interaction.
- **Secure Login**: Built-in user authentication and profile management.

--- 

This README reflects a diabetes prediction project while maintaining the interface and functionality similar to the stock analysis dashboard you provided. Let me know if you'd like to refine any part!
