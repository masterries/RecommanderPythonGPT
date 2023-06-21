# Importieren Sie die erforderlichen Bibliotheken
import streamlit as st
import openai
import pinecone
import os
import sqlite3
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

#####API MANAGEMENT#####
load_dotenv()
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),  
    environment=os.getenv('PINECONE_ENV')  
)
model_name = 'text-embedding-ada-002'
openai.api_key=os.getenv('OPENAI_API_KEY')
index_name = 'langchain-retrieval-augmentation-full'
index = pinecone.Index(index_name=index_name)

#####Menu#####
page = st.sidebar.selectbox("Wählen Sie eine Seite", ["Articles Search", "AI Bot", "MultiLevelRecommendation", "About"])
number_ofRecom1 = st.sidebar.number_input("Number of Recommendations", min_value=1, max_value=10, value=3)
select_option = st.sidebar.selectbox("Recommendations based on", ("content", "title"))
question_mode = st.sidebar.selectbox("Ask Mode on", ("Off", "On"))




def create_embedding(query):

    return openai.Embedding.create(
        input=[query],
        engine=model_name
    )

def query_pinecone_embedding(openai_embedding,number_ofRecom=number_ofRecom1):
    # retrieve from Pinecone
    #is has embedding
    xq = openai_embedding['data'][0]['embedding']
    # get relevant contexts (including the questions)
    return index.query(xq, top_k=number_ofRecom, include_metadata=True)

def pinecone_to_list(res):
    ids = [item['metadata']['o_id'] for item in res['matches']]
    titles = [item['metadata']['title'] for item in res['matches']]
    contents = [getContentArticlebyIDContent(id) for id in ids]
    print(contents)
    items = ["Title = " + title + "\nContent = " + content for title, content in zip(titles, contents)]
    
    return "\n\n---\n\n".join(items)

def pinecone_to_list1(res):

    # get list of retrieved texts
    contexts = [item['metadata']['text'] for item in res['matches']]
    
    # get list of metadata titles
    titles = [item['metadata']['title'] for item in res['matches']]

    items = ["Title = " + title + "\nContent = " + context for title, context in zip(titles, contexts)]
    
    return "\n\n---\n\n".join(items)+"\n\n-----\n\n"

def query_gpt(prompt, message, model):
    return openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": message}
        ]
    ).choices[0].message['content']

prompt = f"""You are Q&A bot. A highly intelligent system that answers
user questions based on the information provided by the user above
each question and references the Title of the information.
If the information can not be found in the information
provided by the user you truthfully say "I don't know".
"""
model = "gpt-3.5-turbo-16k-0613"
def recommend_articles(query):
    embedding = create_embedding(query)
    result = query_pinecone_embedding(embedding)
    print(result)
    open_ai_message = pinecone_to_list1(result)
    return open_ai_message
def search_docs(query, model):
    embedding = create_embedding(query)
    result = query_pinecone_embedding(embedding)
    open_ai_message = pinecone_to_list(result)
    return query_gpt(prompt, open_ai_message, model=model),open_ai_message

def getSearchArticles(query,num_articles):
    embedding = create_embedding(query)
    result = query_pinecone_embedding(embedding,num_articles)
    ids = [item['metadata']['o_id'] for item in result['matches']]
    contents = [get_search_articlesRaw(id) for id in ids]
    combined_df = pd.concat(contents, ignore_index=True)
    
    print(combined_df)
    return combined_df


def recommend_articles1(query, exclude_title):

    if select_option == 'title':
        query = exclude_title

    print(query)
    embedding = create_embedding(query)
    result = query_pinecone_embedding(embedding)

    # Initialisiere eine leere Liste für die Artikel
    articles = []

    # Iteriere durch die Matches im Ergebnis
    for match in result['matches']:
        # Extrahiere o_id, title und score
        o_id = match['metadata']['o_id']
        title = match['metadata']['title']
        score = match['score']
        
        # Überspringe den Artikel, wenn der Titel dem exclude_title entspricht
        if title == exclude_title:
            continue
        
        # Verwende getContentArticlebyID, um den Artikelinhalt und den Titel mit der entsprechenden o_id zu erhalten
        article_info = getContentArticlebyID(o_id)
        content = article_info['content'].values[0]  # get the content string
        author = article_info['author'].values[0]  # get the author string
        date = article_info['date'].values[0]  # get the date string
        publication = article_info['publication'].values[0]  # get the publication string
        
        # Füge den Artikelinhalt und seine Informationen zur Liste der Artikel hinzu
        articles.append({'content': content, 'title': title, 'score': score, 'author': author, 'date': date, 'publication': publication})

    return articles


def getContentArticlebyID(id):
    conn = sqlite3.connect('./all-the-news.db')
    df = pd.read_sql_query(f"SELECT content, title , author, date , publication FROM longform WHERE id = {id}", conn)
    conn.close()
    return df

def getContentArticlebyIDContent(id):
    conn = sqlite3.connect('./all-the-news.db')
    df = pd.read_sql_query(f"SELECT content FROM longform WHERE id = {id}", conn)
    conn.close()
    if df.empty: 
        return '' 
    else: 
        return df['content'].values[0] 
    
def get_search_articlesRaw(o_id):
    conn = sqlite3.connect('./all-the-news.db')
    df = pd.read_sql_query(f"SELECT * FROM longform WHERE  id = {o_id}", conn)
    conn.close()
    return df


import time

def get_random_articles(num_articles):
    conn = sqlite3.connect('./all-the-news.db')
    df = pd.read_sql_query(f"SELECT * FROM longform WHERE content IS NOT NULL AND TRIM(content) <> '' ORDER BY RANDOM() LIMIT {num_articles}", conn)
    conn.close()
    return df


def recommend_articles(row):
    # Only recommend new articles if they haven't been recommended yet, or if a different row is being considered.

    articles = recommend_articles1(row['content'],row['title'])
    # add articles to session state
    st.session_state.current_articles.append(articles)
    showArticles_rec(articles)

# Then in your showArticles_rec function
def showArticles_rec(articles):
    for i, article in enumerate(articles):
        # Check if the content is None or has less than 10 characters
        if article['content'] is None or len(article['content']) < 10:
            continue

        st.subheader(f"{article['title']}")
        col1, col2, col3= st.columns(3)
        col1.text(f"Score: {article['score']}")
        col2.text(f"Author: {(article['author'])}")
        col3.text(f"Date: {(article['date'])}")
        unique_key = f"{article['title']}_{article['score']}"
        

        # Check if the content is a list
        if isinstance(article['content'], list):
            # If so, join it into a single string
            content = ' '.join(article['content'])
        else:
            # If not, use the content as is
            content = article['content']

        # Display the content
        st.text_area("Content", content, height=200 , key =unique_key ) 
        
        #handle_article_selection(article, i)

def create_embedding1(query):
    embedding_response = openai.Embedding.create(
        input=[query],
        engine=model_name
    )
    return embedding_response['data'][0]['embedding']


def render_sidebar():

    for article in st.session_state.selected_articles:
        st.sidebar.text(article['title'])

def handle_article_selection(row, i):
    # Check if article is already selected
    unique_key = f"{row['title']}_{row['author']}"
    if row['title'] in [article['title'] for article in st.session_state.selected_articles]:
        if st.button(f'Deselect article {i}' , key = unique_key):
            st.session_state.selected_articles = [article for article in st.session_state.selected_articles if article['title'] != row['title']]
    else:
        # If not, show "Select" button
        if st.button(f'Select article {i}' , key = unique_key):
            st.session_state.selected_articles.append(row)

def show_articles(df):
    #get length of st.session_state.num_articles
    it = range(st.session_state.num_articles)
    print(it)
    for i in it:
        # Take the i-th record
        row = df.iloc[i]

        # Check if the content is None has less than 10 characters
        if row['content'] is None or len(row['content']) < 10:
            continue

        st.subheader(f"Article {i}:  {row['title']}")
        unique_key = f"{i}-{time.time()}"
        col1, col2, col3 = st.columns(3)
        col1.text(f"Author: {row['author']}")
        col2.text(f"Date: {row['date']}")
        col3.text(f"Publication: {row['publication']}")

        # Check if the content is a list
        if isinstance(row['content'], list):
            # If so, join it into a single string
            content = ' '.join(row['content'])
        else:
            # If not, use the content as is
            content = row['content']
        handle_article_selection(row, i)
        st.text_area("Content", content, height=200)
        if st.button(f'Recommend Articles like this {i}', key = row['title']):
            recommend_articles(row)
        if(question_mode == "On"):
            queryUser = st.text_input(f"Ask Something about Article  {i}", "Summery of Article")
            if st.button(f'Ask Questions {i}'):
                query = "Only Answer Question regarding my Article " + row['title'] + "UserQuestion is" +queryUser+"content is" + content 
                result, open_ai_message = search_docs(query, model=model)
                result = query_gpt(prompt, query, model)
                st.write(result)
        

def recommend_articles_multiple(articles_content_list, exclude_titles):

    embeddings = [create_embedding1(content) for content in articles_content_list]

  
    #print(embeddings)
    avg_embedding = np.mean(np.array(embeddings), axis=0)

    # Create a OpenAIObject embedding with the average embedding
    avg_openai_embedding = {'data': [{'embedding': avg_embedding.tolist()}]}

    # Query the Pinecone DB with the average embedding
    result = query_pinecone_embedding(avg_openai_embedding)


    # Extract articles, excluding those with titles in exclude_titles
    recommended_articles = []
    for match in result['matches']:
        o_id = match['metadata']['o_id']
        title = match['metadata']['title']
        
        score = match['score']

        if title in exclude_titles:
            continue

        article_info = getContentArticlebyID(o_id)
        content = article_info['content'].values[0]  # get the content string
        author = article_info['author'].values[0]  # get the author string
        date = article_info['date'].values[0]  # get the date string
        publication = article_info['publication'].values[0]  # get the publication string
        
        # Füge den Artikelinhalt und seine Informationen zur Liste der Artikel hinzu
        recommended_articles.append({'content': content, 'title': title, 'score': score, 'author': author, 'date': date, 'publication': publication})



    return recommended_articles



def show_selected_articles():
    st.sidebar.header("Selected Articles")
    for article in st.session_state.selected_articles:
        st.sidebar.text(article['title'])

def Seite3():
    st.header("Here are some Random Articles")

    # Slider to select the number of articles
    num_articles = st.slider('Select number of articles', 1, 20, 1)
    userSearch = st.text_input("Articel Search", "Enter your Search here")
    if st.button("Search"):
        df = getSearchArticles(userSearch, num_articles)
        st.session_state.current_articles = df
        st.session_state.num_articles = num_articles
    

    if "current_articles" not in st.session_state or st.session_state.num_articles != num_articles:
        if userSearch == "Enter your Search here" or userSearch == "":
            st.session_state.current_articles = get_random_articles(num_articles)
        else:
            df = getSearchArticles(userSearch, num_articles)
            st.session_state.current_articles = df
            st.session_state.num_articles = num_articles

        st.session_state.num_articles = num_articles
    print(st.session_state.current_articles)

    # Initialize selected_articles if not exists
    if "selected_articles" not in st.session_state:
        st.session_state.selected_articles = []

    show_articles(st.session_state.current_articles)

    show_selected_articles()

    if st.sidebar.button("Recommend Articles based on Selected"):
        # Get contents and titles of selected articles
        selected_articles_content = [article['content'] for article in st.session_state.selected_articles]
        selected_articles_titles = [article['title'] for article in st.session_state.selected_articles]

        if select_option == 'title':
            print('title1')
            recommended_articles = recommend_articles_multiple(selected_articles_content, selected_articles_titles)
        else:
            print('content1')
            recommended_articles = recommend_articles_multiple(selected_articles_titles, selected_articles_titles)
        # Convert list of dictionaries into a DataFrame
        #df = pd.DataFrame(recommended_articles)
        #print(df)

        # Show recommended articles
        showArticles_rec(recommended_articles)
    
    if st.sidebar.button("Clear List"):
        st.session_state.selected_articles = []
        render_sidebar()


    





import streamlit as st





if page == "Articles Search":
    st.header("Welcome to the Article Search")
    if "selected_articles" not in st.session_state:
        st.session_state.selected_articles = []
    num_articles = 1
    userSearch = st.text_input("Articel Search", "Enter your Search here")
    if st.button("Search"):
        st.write("You searched for: ", userSearch)
        st.write("Here are your results:")
        df = getSearchArticles(userSearch,3)
        st.session_state.current_articles = df
        #get number of row of df 
        num_articles = df.shape[0]

        st.session_state.num_articles = num_articles
        show_articles(df)
        #st.write(df)

        if st.button(f'Ask Questions {i}'):
            
            query = "Only Answer Question regarding my Articel " + row['title'] + "UserQuestion is" +queryUser+"content is" + content 
            result, open_ai_message = search_docs(query, model=model)
            result = query_gpt(prompt, query, model)
            st.write(result)

if page == "MultiLevelRecommendation":
    Seite3()








# Inhalt von Seite 2 anzeigen
elif page == "AI Bot":
    st.header("AI Chatbot")
    query = st.text_input("Enter your question")
    boola = False
    if st.checkbox('Show debug information'):
        boola = True
    result, open_ai_message = None, None
    if st.button('Get Answer'):
        result, open_ai_message = search_docs(query, model=model)
        st.write(result)
        if boola:
            st.subheader("Debug Information")
            st.write(open_ai_message)

elif page == "About":
    #load sql data
    st.header("About")
    conn = sqlite3.connect('all-the-news-33.db')
    df = pd.read_sql_query("SELECT * FROM longform Limit 5", conn)
    st.write(df)

