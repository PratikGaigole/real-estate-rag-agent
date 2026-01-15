from langchain_core.prompts import PromptTemplate

# Main QA prompt (replacement for stuff_prompt)
PROMPT = PromptTemplate(
    input_variables=["summaries", "question"],
    template=(
        "You are a helpful assistant for Real Estate research.\n\n"
        "Use the following extracted parts of a long document to answer the question.\n"
        "If you don't know the answer, just say that you don't know.\n\n"
        "{summaries}\n\n"
        "Question: {question}\n"
        "Answer:"
    )
)

# Example document prompt
EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Content: {page_content}\nSource: {source}"
)
