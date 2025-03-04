import src.langchain_helper_1 as lch
import streamlit as st

st.title("Pets Name Generator")

animal_tuple = ("Cat", "Dog", "Cow", "Hamnster")
gender_tuple = ("Male", "Female")

animal_type = st.sidebar.selectbox("What is your pet?", animal_tuple)
pet_gender = st.sidebar.selectbox(f"What gender is your {animal_type.lower()}?", gender_tuple)
pet_color = st.sidebar.text_area(label=f"What color is your {animal_type.lower()}?", max_chars=15)

if pet_color:
    response = lch.generate_pet_name(animal_type=animal_type, pet_color=pet_color, pet_gender=pet_gender)
    st.text(response)