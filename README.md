# Product-Recommendation-System

## Overview
This project implements a content-based recommendation system for products. It uses text-based features, such as product descriptions and tags, to identify and recommend similar products based on what a user has interacted with. Additionally, the system incorporates user review scores and categorization to prioritize high-rated products and ensure diversity in recommendations.

## Features
- **Content-Based Filtering:** Recommends products based on their textual similarity (using `TF-IDF` and cosine similarity).
- **Incorporation of User Reviews:** Prioritizes products with higher average user review scores.
- **Diverse Recommendations:** Ensures a variety of products from different categories are included in the recommendations.
- **Visualization:** Provides visual insights into the distribution of recommended products by category and price, and the similarity scores for recommended products.

## Technologies
- Python 3.x
- `pandas` for data manipulation
- `scikit-learn` for vectorization and cosine similarity computation
- `matplotlib` and `seaborn` for data visualization

## Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/abdghad2025/product-recommendation-system.git
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the necessary datasets and place them in the appropriate directories (e.g., `user_reviews_df.csv`, `product_info_df.csv`).

## Usage

### 1. Data Files:
Ensure that you have the following datasets:
- `product_info_df.csv`: Contains information about products including IDs, names, descriptions, tags, categories, and prices.
- `user_reviews_df.csv`: Contains user reviews with user IDs, product IDs, and review scores.

### 2. Running the Code:
After preparing the datasets, you can run the recommendation system using the following command:

```bash
python recommendation_system.py
