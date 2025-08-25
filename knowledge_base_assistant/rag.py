import os
import json
from knowledge_base_assistant import ingest
from openai import OpenAI

client = OpenAI()

index = ingest.load_index()

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
    """
    A robust wrapper for OpenAI LLM calls that includes comprehensive error
    handling, content validation, and graceful failure.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.choices[0].message.content
        
        if not content or not content.strip() or content.strip().lower() == 'none':
            print("⚠️ LLM returned invalid content. Failing gracefully.")
            return "NO" 
            
        return content.strip()

    except Exception as e:
        print(f"⚠️ LLM call failed with exception: {e}")
        return "⚠️ An error occurred while communicating with the language model."


def answer_question(question, user_language='en', model='gpt-4o-mini'):
    """
    Advanced RAG with a "Search-Then-Decide" router to improve robustness.
    """
    # Step 1: Translate to English
    if user_language != 'en':
        translation_prompt = f'Translate the following user question into a concise English search query: "{question}"'
        english_query = llm(translation_prompt, model='gpt-4o-mini')
        print(f"--> Original Query ({user_language}): {question}")
        print(f"--> Translated English Query: {english_query}")
    else:
        english_query = question

    # Step 2: Content search first
    print("--> Performing content search...")
    search_results = search(english_query)
    
    context_for_llm = ""
    if search_results:
        context_parts = [f"Source: {doc.get('title')}\nContent: {doc.get('text')}" for doc in search_results]
        context_for_llm = "\n\n---\n\n".join(context_parts)
    
    # Step 3: Check relevance
    print("--> Asking Router LLM to check context relevance...")
    router_prompt = f"""
You are a relevance checking model. Your task is to determine if the provided CONTEXT contains enough information to answer the USER'S QUESTION.
Respond with only the single word YES or NO.

USER'S QUESTION: {english_query}

CONTEXT:
---
{context_for_llm or "No context was found."}
---

Is the context relevant to the question? Respond with only YES or NO.
""".strip()

    router_decision = llm(router_prompt, model='gpt-4o-mini')
    print(f"--> Router decision: {router_decision}")

    # Step 4: Act on router's decision
    if "YES" in router_decision.upper():
        print("--> Router found relevant context. Synthesizing answer...")
        # If relevant, proceed with synthesis
        api_context = []
        for doc in search_results:
            api_context.append({
                "type": "content", "title": doc.get("title", "N/A"),
                "section_title": doc.get("section_title", "N/A"),
                "page_number": doc.get("page_number", "N/A"),
                "content": doc.get("text", "")
            })

        synthesis_prompt = f"""
You are an expert assistant. Your task is to synthesize a clear and concise answer to the user's question based *only* on the provided CONTEXT.
Formulate your final answer in {user_language}.

USER'S QUESTION (Original): {question}

CONTEXT:
---
{context_for_llm}
---

Based on the context, here is the synthesized answer in {user_language}:
""".strip()
        
        answer = llm(synthesis_prompt, model=model)

        if answer == "NO":
            print("--> Synthesis failed, returning graceful failure message.")
            failure_messages = {
                'en': "I was unable to synthesize a reliable answer from the provided documents. Please try rephrasing your question.",
                'es': "No pude sintetizar una respuesta confiable a partir de los documentos proporcionados. Por favor, intenta reformular tu pregunta.",
                'it': "Non sono stato in grado di sintetizzare una risposta affidabile dai documenti forniti. Prova a riformulare la tua domanda."
            }
            return failure_messages.get(user_language, failure_messages['en']), api_context
        return answer, api_context

    else:
        # If NOT relevant, check metadata query
        print("--> Router found context irrelevant. Checking for metadata query as fallback...")
        metadata_keywords = ['author', 'authors', 'year', 'published', 'title', 'document id']
        is_metadata_query = any(keyword in english_query.lower() for keyword in metadata_keywords)

        if is_metadata_query:
            print("--> Detected a metadata query. Retrieving metadata.")
            PROJECT_ROOT = os.getenv('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
            data_path = os.path.join(PROJECT_ROOT, 'data', 'data.jsonl')

            if not os.path.exists(data_path):
                return f"⚠️ Metadata file not found: {data_path}", []
            
            with open(data_path, 'rt', encoding='utf-8') as f_in: all_docs = [json.loads(line) for line in f_in]
            
            api_context, papers_metadata = [], {}
            for doc in all_docs:
                meta = doc.get('document_metadata', {})
                title = meta.get('title')
                if title and title not in papers_metadata:
                    papers_metadata[title] = meta
                    api_context.append({"type": "metadata", "title": meta.get("title"), "authors": meta.get("authors"), "year": meta.get("year")})
            
            context_for_llm = json.dumps(list(papers_metadata.values()), indent=2)
            metadata_prompt = f"""
You are a research assistant. Answer the user's question based *only* on the provided METADATA.
Formulate your final answer in {user_language}.

USER'S QUESTION (Original): {question}

METADATA:
---
{context_for_llm}
---

Based on the metadata, here is the answer in {user_language}:
""".strip()
            answer = llm(metadata_prompt, model=model)
            return answer, api_context

        else:
            # Irrelevant AND not metadata
            print("--> Not a metadata query. Returning graceful failure message.")
            failure_messages = {
                'en': "I could not find a specific answer for your question in the provided documents. Please try rephrasing your query.",
                'es': "No pude encontrar una respuesta específica para tu pregunta en los documentos proporcionados. Por favor, intenta reformular tu consulta.",
                'it': "Non sono riuscito a trovare una risposta specifica alla tua domanda nei documenti forniti. Prova a riformulare la tua domanda."
            }
            return failure_messages.get(user_language, failure_messages['en']), []


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