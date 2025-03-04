import src.langchain_helper_2 as lch
import streamlit as st
import textwrap

st.title("YouTube Assitant")

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url = st.sidebar.text_area(
            label="What is the YouTube video URL?",
            max_chars=50
            )
        query = st.sidebar.text_area(
            label="Ask me about the video?",
            max_chars=60,
            key="query"
            )
        # openai_api_key = st.sidebar.text_input(
        #     label="OpenAI API Key",
        #     key="langchain_search_api_key_openai",
        #     max_chars=100,
        #     type="password"
        #     )
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        submit_button = st.form_submit_button(label='Submit')

if query and youtube_url:
    # if not openai_api_key:
    #     st.info("Please add your OpenAI API key to continue.")
    #     st.stop()
    # else:
    db = lch.create_vector_db_from_yt_url(youtube_url)
    response, docs = lch.get_response_from_query(db, query)
    st.subheader("Answer:")
    st.text(textwrap.fill(response, width=85))