from contextlib import contextmanager
import os
import json

import hmac
import streamlit as st
st.set_page_config(page_title = 'Hunome RAG prototype', page_icon = '🔥', layout = 'wide')
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, DeclarativeBase, mapped_column
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    VARCHAR,
    String,
    DateTime,
    func
)

from infisical import InfisicalClient

from huggingface_hub import InferenceClient
from openai import OpenAI

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def check_password():
    '''Returns `True` if the user had the correct password.'''

    def password_entered():
        '''Checks whether a password entered by the user is correct.'''
        if hmac.compare_digest(st.session_state['password'], st.secrets['password']):
            st.session_state['password_correct'] = True
            del st.session_state['password']  # Don't store the password.
        else:
            st.session_state['password_correct'] = False

    # Return True if the passward is validated.
    if st.session_state.get('password_correct', False):
        return True

    # Show input for password.
    st.text_input(
        'Password', type='password', on_change=password_entered, key='password'
    )
    if 'password_correct' in st.session_state:
        st.error('😕 Password incorrect')
    return False

@st.cache_data
def get_secrets():
    secrets = {}
    if_client = InfisicalClient(token = os.environ.get('INFISICAL_TOKEN'))

    secrets['DWH_USER']   = if_client.get_secret('DWH_PG_USER').secret_value
    secrets['DWH_PW']     = if_client.get_secret('DWH_PG_PW').secret_value
    secrets['DWH_HOST']   = if_client.get_secret('DWH_PG_HOST').secret_value
    secrets['DWH_DBNAME'] = if_client.get_secret('DWH_PG_DBNAME').secret_value

    secrets['HF_API_KEY'] = if_client.get_secret('HF_API_KEY').secret_value
    secrets['OPENAI_API_KEY'] = if_client.get_secret('OPENAI_API_KEY').secret_value

    return secrets

secrets = get_secrets()

DWH_URL = f'postgresql+psycopg2://{secrets["DWH_USER"]}:{secrets["DWH_PW"]}@{secrets["DWH_HOST"]}:5432/{secrets["DWH_DBNAME"]}'
os.environ['OPENAI_API_KEY'] = secrets['OPENAI_API_KEY']
openai_client = OpenAI()

hf_client = InferenceClient(token = secrets['HF_API_KEY'])

class Base(DeclarativeBase):
    pass

class SparkEmbeddings(Base):
    __tablename__ = "spark_embeddings"
    spark_id = mapped_column(VARCHAR(36), primary_key = True)
    map_id = mapped_column(VARCHAR(36))
    entity_updated = mapped_column(DateTime)
    title = mapped_column(String)
    fulltext = mapped_column(String)
    embedding = mapped_column(Vector(384))

@contextmanager
def get_session(db_url) -> Session:
    engine = create_engine(db_url, echo = True)
    session = Session(engine)
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()
        engine.dispose()

def llm(prompt, model = 'gpt-3.5-turbo', client = openai_client):
    return client.chat.completions.create(
        model = model,
        messages = prompt,
        stream = False,
        seed = 42,
        temperature = 0.2,
    )

@st.cache_data
def load_sparkmaps() -> dict[str, list[str, int]]:
    with get_session(DWH_URL) as dbs:
        cnt_stmt = dbs.query(SparkEmbeddings.map_id, func.count(SparkEmbeddings.spark_id).label('count')).group_by(SparkEmbeddings.map_id).subquery()
        sparkmaps = dbs.query(cnt_stmt, SparkEmbeddings.title).\
            join(SparkEmbeddings, SparkEmbeddings.spark_id == cnt_stmt.c.map_id).\
            order_by(cnt_stmt.c.count.desc())
    return {sm.map_id: [sm.title, sm.count] for sm in sparkmaps}

def format_spark_map_select(sparkmaps: dict[str, list[str, int]], sparkmap_id: str) -> str:
    return sparkmaps[sparkmap_id][0] + f' ({sparkmaps[sparkmap_id][1]})'

@st.cache_data
def get_sparks(query: str, sparkmap_ids: list[str], query_size: int):
    q_embedding = hf_client.feature_extraction(query, model = 'BAAI/bge-small-en-v1.5')
    with get_session(DWH_URL) as dbs:
        query_results = dbs.query(SparkEmbeddings.spark_id, SparkEmbeddings.title, SparkEmbeddings.fulltext)
        if len(sparkmap_ids):
            query_results = query_results.\
            filter(SparkEmbeddings.map_id.in_(sparkmap_ids))
        query_results = query_results.\
            order_by(SparkEmbeddings.embedding.l2_distance(q_embedding)).\
            limit(query_size)
    return [{
        'source': r.spark_id,
        'title': r.title,
        'content': r.fulltext,
    } for r in query_results]

@st.cache_data
def generate_response(query: str, sparks: list[dict[str, str]], gpt_model: str):
    prompt = [
        {"role": "system", "content": "You're an honest, helpful and polite assistent answering questions from the provided context and only from context. If the context is not enough, you honestly admit that there is not enough data to answer the question."},
        {"role": "system", "content": "The context is provided as JSON array of objects with title, content and source keys, where content is the text and source is the source of the text. For each and every generated sentence you must provide the source of the sentence."},
        {"role": "system", "content": "Context: " + json.dumps(sparks)},
        {"role": "user", "content": query}
    ]
    logging.debug(f'Prompt to LLM: {prompt}')
    return llm(prompt, model = gpt_model)

def process(query: str, sparkmap_ids: list[str], query_size: int, gpt_model: str):
    if len(query) < 20:
        st.subheader('Please provide a longer question')
        return
    sparks = get_sparks(query, sparkmap_ids, query_size)
    logging.debug(f'Found {len(sparks)} sparks')
    response = generate_response(query, sparks, gpt_model)
    if len(sparks) == 0:
        st.subheader('No relevant sparks found in selected SparkMaps')
        return
    st.subheader('Use these sources to answer your question')
    for spark in sparks:
        st.markdown(f'* [{spark["title"]} ({spark["source"]})](https://platform.hunome.com/sparkmap/view-spark/{spark["source"]})')
    st.subheader('Response')
    # for part in response:
    #     st.write(part.choices[0].delta.content or "", end = "")
    st.write(response.choices[0].message.content)

if __name__ == '__main__':
    if not check_password():
        st.stop()  # Do not continue if check_password is not True.

    sparkmaps = load_sparkmaps()
    sparkmap_ids = st.sidebar.multiselect(
        'Select Spark Map(s)',
        sparkmaps.keys(),
        format_func = lambda x: format_spark_map_select(sparkmaps, x)
    )

    query_size = st.sidebar.slider('Number of sparks to query', 2, 7, 3, 1, on_change = None, args = None, key = None)
    gpt_model = st.sidebar.selectbox(
        'GPT Model', ('gpt-3.5-turbo', 'gpt-4-1106-preview')
    )
    query = st.sidebar.text_area('Your quetion')

    # submit_button = st.sidebar.button(
    #     'Make a query',
    #     on_click = process,
    #     args = (query, sparkmap_ids, query_size, gpt_model)
    # )
    process(query, sparkmap_ids, query_size, gpt_model)
