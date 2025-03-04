# from langchain.document_loaders import YoutubeLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()
video_url = "https://www.youtube.com/watch?v=-Osca2Zax4Y" # "https://www.youtube.com/watch?v=-moW9jvvMr4"

def create_vector_db_from_yt_url(video_url:str)->FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(documents=docs, embedding=embeddings)

    return db 


def get_response_from_query(db, query, k=4):
    # text-davinci can handle 4097 tokens

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model = "gpt-3.5-turbo-instruct")
    prompt_template_name = PromptTemplate(
                                            input_variables=["question", "docs"],
                                            template="""
                                            You are a helpful assistant that that can answer questions about youtube videos 
                                            based on the video's transcript.
                                            
                                            Answer the following question: {question}
                                            By searching the following video transcript: {docs}
                                            
                                            Only use the factual information from the transcript to answer the question.
                                            
                                            If you feel like you don't have enough information to answer the question, say "I don't know".
                                            
                                            Your answers should be verbose and detailed.
                                            """
                                        )
    name_chain = prompt_template_name | llm | StrOutputParser()
    response = name_chain.invoke({"question": query, "docs":docs_page_content})
    # response = response.replace('\n', '')
    
    return response, docs

if __name__ == "__main__":
    print(create_vector_db_from_yt_url(video_url))