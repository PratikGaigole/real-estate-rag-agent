import streamlit as st
from rag import process_urls, generate_answer

st.set_page_config(page_title="Real Estate Research Tool")
st.title("🏠 Real Estate Research Tool")

# Sidebar inputs
st.sidebar.header("Enter URLs")
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

status_placeholder = st.empty()

# Process URLs button
if st.sidebar.button("Process URLs"):
    urls = [url for url in (url1, url2, url3) if url.strip()]

    if not urls:
        status_placeholder.error("You must provide at least one valid URL")
    else:
        for status in process_urls(urls):
            status_placeholder.info(status)

st.divider()

# Question input (MAIN PAGE, not placeholder)
query = st.text_input("Ask a question about the properties")

if query:
    try:
        answer, sources = generate_answer(query)

        st.subheader("Answer")
        st.write(answer)

        if sources:
            st.subheader("Sources")
            for source in sources:
                st.write(source)

    except RuntimeError:
        st.error("Please process URLs before asking a question")
