import pandas as pd
import streamlit as st
import pickle
from sklearn.metrics.pairwise import linear_kernel
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Set page configuration
st.set_page_config(
    page_title="Recipe Recommendation App",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# Rest of your Streamlit app code...

# Display the about section when a specific condition is met


# Load the model
with open('model.pkl', 'rb') as file:
    nn_model = pickle.load(file)

# Load the TF-IDF vectorizer
with open('vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Load the recipes data
data = pd.read_csv('pre_processed.csv')   

ps = PorterStemmer()

def transform_text(ingredients, total_time):
    # Process ingredients and total time as needed
    # For example, you can concatenate them into a single string
    text = f"{ingredients} {total_time} minutes"
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))

    return ' '.join(y)    

# Define function to get recipe recommendations
def get_recipe_recommendations(ingredients, total_time):
    # Process input
    input_text = transform_text(ingredients, total_time)

    # Transform input text using the trained TF-IDF vectorizer
    input_tfidf = tfidf_vectorizer.transform([input_text])

    # Find nearest neighbors
    _, indices = nn_model.kneighbors(input_tfidf)

    # Return recipe information
    recommendations = []
    for i in indices[0]:
        recommendations.append({
            'recipe_name': data.iloc[i]['recipe_name'],
            'ingredients': data.iloc[i]['ingredients'],
            'directions': data.iloc[i]['directions'],
            'total_time': data.iloc[i]['total_time'],
            'nutrition': data.iloc[i]['nutrition'],
            'servings': data.iloc[i]['servings'],
            'img_src': data.iloc[i]['img_src']
        })

    return recommendations

# Streamlit App
def main():
    # Apply custom CSS for styling
    st.markdown("""
        <style>
            body {
                background-color: #f5f5f5; /* Light gray background color */
                color: #333;
            }
            .stApp {
                max-width: 800px;
                margin: 0 auto;
            }
            .title {
                font-size: 36px;
                font-weight: bold;
                color: #e74c3c; /* Red title color */
                padding-bottom: 20px;
            }
            .button {
                background-color: #3498db; /* Blue button color */
                color: white;
                padding: 10px 20px;
                font-size: 18px;
                border-radius: 5px;
                cursor: pointer;
                text-align: center;
            }
            .subheader {
                font-size: 24px;
                font-weight: bold;
                color: #27ae60; /* Green subheader color */
                padding-top: 20px;
            }
            .recipe-info {
                font-size: 18px;
                color: #5F6F52;
                padding: 10px 0;
            }
            .recipe-image {
                border-radius: 8px;
                margin-bottom: 10px;
            }
            .input-label {
                font-size: 18px;
                font-weight: bold;
                color: #e74c3c; /* Red input label color */
                padding-top: 10px;
            }
            .input-field {
                font-size: 20px;
                color: #2c3e50;  /* Dark gray input field color */
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='color: #black; padding-bottom: 30px'>Recipe Recommendation App</h1>", unsafe_allow_html=True)
    
    st.markdown('<hr style="border-top: 7px solid green; margin: -20px 0 80px 0; width: 90%;">', unsafe_allow_html=True)

# User Input
    st.markdown('<p class="input-label" style="padding-bottom: 0px;"><b>Enter Ingredients (comma-separated)</b></p>', unsafe_allow_html=True)
    ingredients = st.text_area("", "rice, brinjal", key='ingredients', height=100)
    st.markdown('<p class="input-label" style="padding-bottom: 0px;"><b>Enter Total Time (minutes)</b></p>', unsafe_allow_html=True)
    total_time = st.slider("", 0, 120, 60, key='total_time')


    if st.button("Get Recommendations"):
        recommendations = get_recipe_recommendations(ingredients, total_time)
        st.subheader("Top 10 Recipes Similar to Your Input:")

        for i, recipe_info in enumerate(recommendations, start=1):
            st.markdown(f"<div class='recipe-info' style='color: #e74c3c;'>{i}. <strong>{recipe_info['recipe_name']}</strong></div>", unsafe_allow_html=True)
            st.markdown(
                f'<div style="text-align: center; padding: 20px;"><img src="{recipe_info["img_src"]}" alt="Recipe Image" style="max-width: 80%; max-height: 350px"></div>',
                unsafe_allow_html=True
            )
            st.write(f"<div class='recipe-info'><strong>Ingredients:</strong> {recipe_info['ingredients']}</div>", unsafe_allow_html=True)
            st.write(f"<div class='recipe-info'><strong>Directions:</strong> {recipe_info['directions']}</div>", unsafe_allow_html=True)
            st.write(f"<div class='recipe-info'><strong>Total Time:</strong> {recipe_info['total_time']} minutes</div>", unsafe_allow_html=True)
            st.write(f"<div class='recipe-info'><strong>Nutrition:</strong> {recipe_info['nutrition']}</div>", unsafe_allow_html=True)
            st.write(f"<div class='recipe-info'><strong>Serving:</strong> {recipe_info['servings']}</div>", unsafe_allow_html=True)
            st.write("")  # Add an empty line for better readability

# About Section
st.sidebar.title("About")
st.sidebar.info(
    """
    This Recipe Recommendation App is designed to help you discover new and exciting recipes based on your preferences.
    Simply enter the ingredients you have, set the total time you want to spend, and let the app provide you with personalized recipe recommendations.

    **How it Works:**
    - Enter your ingredients in the specified format (comma-separated).
    - Adjust the total time slider to indicate your preferred cooking time.
    - Click the "Get Recommendations" button, and the app will display the top 10 recipes that match your input.

    **Note:**
    This app uses a content-based recommendation system to analyze and suggest recipes based on your input.

    Enjoy exploring new recipes and happy cooking!
    """
)

if __name__ == "__main__":
    main()
