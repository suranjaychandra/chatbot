# app.py
import streamlit as st
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryMemory
from dotenv import load_dotenv
import os
import requests
import wikipedia

# Load environment variables
load_dotenv()

# Environment Variables
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_base = os.getenv("AZURE_OPENAI_API_BASE")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
news_api_key = os.getenv("NEWSAPI_KEY")

# Get Wikipedia Summary
def get_wikipedia_summary(query, sentences=3):
    redirects = {
        "ahmedabad plane crash": "Air India Flight 171"
    }
    title = redirects.get(query.lower(), query)
    try:
        page = wikipedia.page(title, auto_suggest=False, redirect=True)
        return page.summary[:1000]
    except Exception as e:
        return f"(Wikipedia error: {e})"

# Get Latest News
def get_latest_news(query, api_key, language="en", page_size=3):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": language,
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key,
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data["status"] == "ok" and data["totalResults"] > 0:
            news_items = data["articles"]
            summary = "\n\n".join([f"🔹 **{item['title']}**\n{item['description']}" for item in news_items])
            return summary
        else:
            return "(No recent news found for this topic.)"
    except Exception as e:
        return f"(News API error: {str(e)})"

# Load LLM chain
@st.cache_resource
def load_chain():
    llm = AzureChatOpenAI(
        deployment_name=deployment_name,
        api_version=api_version,
        api_key=api_key,
        azure_endpoint=api_base,
        model_name="gpt-4",
        temperature=0.9,
        top_p=0.9,
        max_tokens=300,
    )

    prompt = PromptTemplate.from_template(
        """You are a helpful programming assistant.

When the user asks to write code, always:
- Provide the complete code block
- Use proper indentation and formatting
- Include the correct opening and closing HTML/JS/CSS/script tags
- Make the response structured, readable, and production-ready
- Optionally include helpful comments or explain key parts if needed

User Question: {question}

Your Response:
Wikipedia: {wiki_info}
Recent News: {news_info}
Chat history: {chat_history}

Question: {question}
Answer:"""
    )
 

    memory = ConversationSummaryMemory(
        llm=llm,
        max_token_limit=1000,
        return_messages=True,
        input_key="question",
        memory_key="chat_history",
    )

    return LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
    )

chain = load_chain()
st.title("🧠 GPT Chatbot")  # ✅ top-level title
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        wiki_info = get_wikipedia_summary(prompt)
        news_info = get_latest_news(prompt, news_api_key)

        result = chain.invoke({
            "question": prompt,
            "wiki_info": wiki_info,
            "news_info": news_info
        })

        response = result["text"]

        for chunk in response.split():
            full_response += chunk + " "
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": response})
