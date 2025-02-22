import streamlit as st
import json
import os
import re
import string
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
import pickle
import numpy as np
import pytesseract
import PIL
import demoji

session_state = st.session_state
if "user_index" not in st.session_state:
    st.session_state["user_index"] = 0
def extract_emojis(text):
    emojis = re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', text)
    return emojis
def ocr_core(img):
    text = pytesseract.image_to_string(PIL.Image.open(img))
    return text
def preprocess_text(text):

    text = re.sub(r'http\S+', '', text)

    text = re.sub("[^a-zA-Z#@]", " ", text)

    text = text.lower()

    words = word_tokenize(text)


    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    exclude =string.punctuation
    words = [word for word in words if word not in exclude]


    lemmatizer = WordNetLemmatizer()
    tagged_words = pos_tag(words)
    lemmatized_words = []
    for word, tag in tagged_words:
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        elif tag.startswith('JJ'):
            pos = 'a'
        else:
            pos = 'n'
        lemmatized_words.append(lemmatizer.lemmatize(word, pos))
    text = ' '.join(lemmatized_words)
    return text



def predict_single_text(text, tokenizer, model):
    classes =['age',
            'ethnicity',
            'gender',
            'not_cyberbullying',
            'other_cyberbullying',
            'religion']
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    encoding = tokenizer(text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)

    return classes[preds.item()]

def signup(json_file_path="data.json"):
    st.title("Signup Page")
    with st.form("signup_form"):
        st.write("Fill in the details below to create an account:")
        name = st.text_input("Name:")
        email = st.text_input("Email:")
        age = st.number_input("Age:", min_value=0, max_value=120)
        sex = st.radio("Sex:", ("Male", "Female", "Other"))
        password = st.text_input("Password:", type="password")
        confirm_password = st.text_input("Confirm Password:", type="password")

        if st.form_submit_button("Signup"):
            if password == confirm_password:
                user = create_account(name, email, age, sex, password, json_file_path)
                session_state["logged_in"] = True
                session_state["user_info"] = user
                session_state["page"] = "Dashboard"
            else:
                st.error("Passwords do not match. Please try again.")

def check_login(username, password, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        for user in data["users"]:
            if user["email"] == username and user["password"] == password:
                session_state["logged_in"] = True
                session_state["user_info"] = user
                st.success("Login successful!")
                return user

        st.error("Invalid credentials. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error checking login: {e}")
        return None
def initialize_database(json_file_path="data.json"):
    try:

        if not os.path.exists(json_file_path):

            data = {"users": []}
            with open(json_file_path, "w") as json_file:
                json.dump(data, json_file)
    except Exception as e:
        print(f"Error initializing database: {e}")
        
def create_account(name, email, age, sex, password, json_file_path="data.json"):
    try:

        if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0:
            data = {"users": []}
        else:
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)

        user_info = {
            "name": name,
            "email": email,
            "age": age,
            "sex": sex,
            "password": password,

        }
        data["users"].append(user_info)


        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        st.success("Account created successfully! You can now login.")
        return user_info
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        st.error(f"Error creating account: {e}")
        return None

def login(json_file_path="data.json"):
    st.title("Login Page")
    username = st.text_input("Email:")
    password = st.text_input("Password:", type="password")

    login_button = st.button("Login")

    if login_button:
        user = check_login(username, password, json_file_path)
        if user is not None:
            session_state["logged_in"] = True
            session_state["user_info"] = user
            session_state["page"] = "Dashboard"
        else:
            st.error("Invalid credentials. Please try again.")

def get_user_info(email, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            for user in data["users"]:
                if user["email"] == email:
                    return user
        return None
    except Exception as e:
        st.error(f"Error getting user information: {e}")
        return None


def render_dashboard(user_info, json_file_path="data.json"):
    try:
        st.title(f"Welcome to the Dashboard, {user_info['name']}!")
        st.subheader("User Information:")
        st.write(f"Name: {user_info['name']}")
        st.write(f"Sex: {user_info['sex']}")
        st.write(f"Age: {user_info['age']}")

    except Exception as e:
        st.error(f"Error rendering dashboard: {e}")
    
def preprocess_text(text):

    text = re.sub(r'http\S+', '', text)

    text = re.sub("[^a-zA-Z#@]", " ", text)

    text = text.lower()

    words = word_tokenize(text)


    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    exclude =string.punctuation
    words = [word for word in words if word not in exclude]


    lemmatizer = WordNetLemmatizer()
    tagged_words = pos_tag(words)
    lemmatized_words = []
    for word, tag in tagged_words:
        if tag.startswith('NN'):  
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        elif tag.startswith('JJ'): 
            pos = 'a'
        else:
            pos = 'n'  
        lemmatized_words.append(lemmatizer.lemmatize(word, pos))
    text = ' '.join(lemmatized_words)
    return text
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits  
def main(json_file_path="data.json"):
    st.sidebar.title("Cyber Bullyig Identification")
    page = st.sidebar.radio(
        "Go to", ("Signup/Login", "Dashboard", "Cyber Bullying Identification"))
    st.session_state["page"] = page

    if page == "Signup/Login":
        st.title("Signup/Login Page")
        login_or_signup = st.radio(
            "Select an option", ("Login", "Signup"), key="login_signup"
        )
        if login_or_signup == "Login":
            login(json_file_path)
        else:
            signup(json_file_path)

    elif page == "Dashboard":
        if session_state.get("logged_in"):
            render_dashboard(session_state["user_info"])
        else:
            st.warning("Please login/signup to view the dashboard.")

    elif page == "Cyber Bullying Identification":
        if session_state.get("logged_in"):
            st.title("Cyber Bullyig Identification")

            load_tokenizer = BertTokenizer.from_pretrained("tokenizer_dir")
            loaded_model = BERTClassifier(bert_model_name='bert-base-uncased', num_classes=6)
            loaded_model.load_state_dict(torch.load('bert_model.pth', map_location=torch.device('cpu')))
            choice = st.radio("Select the type of content", ("Image", "Text"))
            if choice == "Image":
                uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg"])
                if uploaded_file:
                    image_path = f"temp_image_{session_state['user_index']}.png"
                    with open(image_path, "wb") as file:
                        file.write(uploaded_file.getvalue())
                    st.image(
                        uploaded_file, caption="Image", use_column_width=True
                    )
                    user_input = ocr_core(image_path)
                    if st.button('Submit'):
                        emojis = extract_emojis(user_input)
                        text = preprocess_text(user_input)
                        emojis_description = ' '.join(demoji.findall(' '.join(emojis)).values()) if emojis else ''
                        final_text = text +' '+ emojis_description
                        prediction = predict_single_text(text, load_tokenizer, loaded_model)
                        if prediction.lower() == "not_cyberbullying":
                            st.markdown("<div style='color:green; font-size:20px;'>It is not a CyberBullying</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='color:red; font-size:20px;'>It is a CyberBullying<br> Type : {prediction}</div>".format(prediction=prediction.title()), unsafe_allow_html=True)
                    os.remove(image_path)
            else:
                user_input = st.text_area("Enter the text to classify")
                if st.button('Submit'):
                    emojis = extract_emojis(user_input)
                    text = preprocess_text(user_input)
                    emojis_description = ' '.join(demoji.findall(' '.join(emojis)).values()) if emojis else ''
                    final_text = text +' '+ emojis_description
                    prediction = predict_single_text(text, load_tokenizer, loaded_model)
                    if prediction.lower() == "not_cyberbullying":
                        st.markdown("<div style='color:green; font-size:20px;'>It is not a CyberBullying</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='color:red; font-size:20px;'>It is a CyberBullying<br> Type ---> {prediction}</div>".format(prediction=prediction.title()), unsafe_allow_html=True)        
        else:
            st.warning("Please login/signup to use app!!")           
    else:
        st.warning("Please login/signup to use app!!")
            



if __name__ == "__main__":
    initialize_database()
    main()
