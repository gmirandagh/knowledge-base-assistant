import os
import json
import ingest
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# Load or build index
index = ingest.load_index()

# --- Helper functions (no changes here) ---
def search(query, filter_dict=None):
    """Performs a keyword search with optional filtering."""
    if filter_dict is None:
        filter_dict = {}
    boost = {'title': 3.0, 'section_title': 1.5}
    results = index.search(
        query=query,
        filter_dict=filter_dict,
        boost_dict=boost,
        num_results=7
    )
    return results

def llm(prompt, model='gpt-4o-mini'):
    """Wrapper for OpenAI LLM calls with error handling."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ LLM call failed: {e}"

# --- Main RAG Entry Point (Refactored and Improved) ---

def answer_question(question, user_language='en', model='gpt-4o-mini'):
    """
    Advanced, hybrid RAG function that handles multiple languages robustly.
    1. Translates the query to English.
    2. Checks for metadata keywords in the *English* query.
    3. Executes the correct path (metadata or content search).
    4. Generates the final answer in the user's original language.
    """
    # Step 1: Translate query to English to standardize the logic
    if user_language != 'en':
        translation_prompt = f'Translate the following user question into a concise English search query: "{question}"'
        english_query = llm(translation_prompt, model='gpt-4o-mini')
        print(f"--> Original Query ({user_language}): {question}")
        print(f"--> Translated English Query: {english_query}")
    else:
        english_query = question

    # Step 2: Perform metadata check on the standardized English query
    metadata_keywords = ['author', 'authors', 'year', 'published', 'title', 'document id']
    is_metadata_query = any(keyword in english_query.lower() for keyword in metadata_keywords)
    
    api_context = []
    answer = "No information found."

    if is_metadata_query:
        print("--> Detected a metadata query. Retrieving metadata.")
        # This path is for questions about authors, years, etc.
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        data_path = os.path.join(PROJECT_ROOT, 'data', 'data.jsonl')

        if not os.path.exists(data_path):
            return f"⚠️ Metadata file not found: {data_path}", []

        with open(data_path, 'rt', encoding='utf-8') as f_in:
            all_docs = [json.loads(line) for line in f_in]

        papers_metadata = {}
        for doc in all_docs:
            meta = doc.get('document_metadata', {})
            title = meta.get('title')
            if title and title not in papers_metadata:
                papers_metadata[title] = meta
                api_context.append({"type": "metadata", **meta})

        context_for_llm = json.dumps(list(papers_metadata.values()), indent=2)

        final_prompt = f"""
You are a research assistant. Answer the user's question based *only* on the provided METADATA.
Formulate your final answer in {user_language}.

USER'S QUESTION (Original): {question}
METADATA:
---
{context_for_llm}
---
FINAL ANSWER (in {user_language}):
""".strip()
        answer = llm(final_prompt, model=model)

    else:
        print("--> Performing standard content search.")
        # This path is for questions about the content of documents
        search_results = search(english_query)
        if not search_results:
            # Create a user-friendly message in their language
            if user_language == 'es':
                answer = "No pude encontrar información relevante en los documentos para esa consulta."
            elif user_language == 'it':
                answer = "Non sono riuscito a trovare informazioni pertinenti nei documenti per quella query."
            else:
                answer = "I could not find any relevant information in the documents for that query."
            return answer, []

        context_for_llm_parts = []
        for doc in search_results:
            api_context.append({"type": "content", **doc})
            context_for_llm_parts.append(f"Source: {doc.get('title')}\nContent: {doc.get('text')}")
        
        context_for_llm = "\n\n---\n\n".join(context_for_llm_parts)

        final_prompt = f"""
You are an expert assistant. Answer the user's question based *only* on the provided CONTEXT.
Formulate your final answer in {user_language}.

USER'S QUESTION (Original): {question}
CONTEXT:
---
{context_for_llm}
---
FINAL ANSWER (in {user_language}):
""".strip()
        answer = llm(final_prompt, model=model)

    return answer, api_context

# # Simple CLI in English
# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) < 2:
#         print("Usage: python rag.py '<your question here>'")
#         sys.exit(1)
#     query = sys.argv[1]
#     print(f"\n🔍 Query: {query}\n")
#     answer, context = answer_question(query)
#     print("\n💡 Answer:\n")
#     print(answer)

# Multilanguage CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Query the RAG pipeline directly from your terminal.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "query",
        type=str,
        help="The question you want to ask, enclosed in quotes."
    )

    # Optional '--lang' flag for language selection
    parser.add_argument(
        "--lang",
        type=str,
        default='en',
        choices=['en', 'es', 'it'],
        help="The language of the query.\n"
             "  'en' for English (default)\n"
             "  'es' for Spanish\n"
             "  'it' for Italian"
    )

    args = parser.parse_args()

    print(f"\n🔍 Query: {args.query} (Language: {args.lang})\n")
    answer, context = answer_question(args.query, user_language=args.lang)

    print("\n💡 Answer:\n")
    print(answer)