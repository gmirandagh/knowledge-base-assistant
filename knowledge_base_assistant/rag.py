import os
import json
from knowledge_base_assistant import ingest
from openai import OpenAI
from time import time

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


def llm(prompt, model='gpt-4o-mini', return_stats=False):
    """
    A robust wrapper for OpenAI LLM calls that includes comprehensive error
    handling, content validation, and graceful failure.
    
    Args:
        prompt: The prompt to send to the LLM
        model: The model to use
        return_stats: If True, returns (content, token_stats), otherwise just content
    
    Returns:
        If return_stats=False: content string
        If return_stats=True: tuple (content, token_stats)
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.choices[0].message.content
        
        token_stats = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        if not content or not content.strip() or content.strip().lower() == 'none':
            print("⚠️ LLM returned invalid content. Failing gracefully.")
            if return_stats:
                return "NO", token_stats
            return "NO"
            
        answer = content.strip()
        
        if return_stats:
            return answer, token_stats
        return answer

    except Exception as e:
        print(f"⚠️ LLM call failed with exception: {e}")
        error_msg = "⚠️ An error occurred while communicating with the language model."
        if return_stats:
            return error_msg, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return error_msg


evaluation_prompt_template = """
You are an expert evaluator for a RAG system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


def evaluate_relevance(question, answer):
    """
    Evaluates the relevance of an answer to a question using LLM.
    
    Args:
        question: The original question
        answer: The generated answer
        
    Returns:
        tuple: (evaluation_dict, token_stats)
    """
    prompt = evaluation_prompt_template.format(question=question, answer=answer)
    evaluation, tokens = llm(prompt, model="gpt-4o-mini", return_stats=True)

    try:
        json_eval = json.loads(evaluation)
        return json_eval, tokens
    except json.JSONDecodeError:
        result = {"Relevance": "UNKNOWN", "Explanation": "Failed to parse evaluation"}
        return result, tokens


def calculate_openai_cost(model, tokens):
    """
    Calculates the cost of OpenAI API calls based on token usage.
    
    Args:
        model: The model used
        tokens: Token statistics dict with prompt_tokens and completion_tokens
        
    Returns:
        float: Cost in USD
    """
    openai_cost = 0

    if model == "gpt-4o-mini":
        # Pricing as of Jan 2025 - update as needed
        openai_cost = (
            tokens["prompt_tokens"] * 0.00015 + tokens["completion_tokens"] * 0.0006
        ) / 1000
    elif model == "gpt-4o":
        # Add pricing for other models as needed
        openai_cost = (
            tokens["prompt_tokens"] * 0.0025 + tokens["completion_tokens"] * 0.01
        ) / 1000
    else:
        print(f"Warning: Model {model} not recognized. Cost calculation may be inaccurate.")

    return openai_cost


def answer_question(question, user_language='en', model='gpt-4o-mini', evaluate=False):
    """
    Advanced RAG with a "Search-Then-Decide" router to improve robustness.
    
    Args:
        question: The user's question
        user_language: Language for the response ('en', 'es', 'it')
        model: The LLM model to use
        evaluate: If True, includes evaluation metrics in response
        
    Returns:
        If evaluate=False: tuple (answer, api_context)
        If evaluate=True: tuple (answer, api_context, metrics)
    """
    start_time = time()
    total_cost = 0
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    # Step 1: Translate to English
    if user_language != 'en':
        translation_prompt = f'Translate the following user question into a concise English search query: "{question}"'
        if evaluate:
            english_query, trans_tokens = llm(translation_prompt, model='gpt-4o-mini', return_stats=True)
            total_cost += calculate_openai_cost('gpt-4o-mini', trans_tokens)
            for key in total_tokens:
                total_tokens[key] += trans_tokens[key]
        else:
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

    if evaluate:
        router_decision, router_tokens = llm(router_prompt, model='gpt-4o-mini', return_stats=True)
        total_cost += calculate_openai_cost('gpt-4o-mini', router_tokens)
        for key in total_tokens:
            total_tokens[key] += router_tokens[key]
    else:
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
        
        if evaluate:
            answer, synth_tokens = llm(synthesis_prompt, model=model, return_stats=True)
            total_cost += calculate_openai_cost(model, synth_tokens)
            for key in total_tokens:
                total_tokens[key] += synth_tokens[key]
        else:
            answer = llm(synthesis_prompt, model=model)

        if answer == "NO":
            print("--> Synthesis failed, returning graceful failure message.")
            failure_messages = {
                'en': "I was unable to synthesize a reliable answer from the provided documents. Please try rephrasing your question.",
                'es': "No pude sintetizar una respuesta confiable a partir de los documentos proporcionados. Por favor, intenta reformular tu pregunta.",
                'it': "Non sono stato in grado di sintetizzare una risposta affidabile dai documenti forniti. Prova a riformulare la tua domanda."
            }
            answer = failure_messages.get(user_language, failure_messages['en'])

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
                answer = f"⚠️ Metadata file not found: {data_path}"
                api_context = []
            else:
                with open(data_path, 'rt', encoding='utf-8') as f_in: 
                    all_docs = [json.loads(line) for line in f_in]
                
                api_context, papers_metadata = [], {}
                for doc in all_docs:
                    meta = doc.get('document_metadata', {})
                    title = meta.get('title')
                    if title and title not in papers_metadata:
                        papers_metadata[title] = meta
                        api_context.append({
                            "type": "metadata", 
                            "title": meta.get("title"), 
                            "authors": meta.get("authors"), 
                            "year": meta.get("year")
                        })
                
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
                
                if evaluate:
                    answer, meta_tokens = llm(metadata_prompt, model=model, return_stats=True)
                    total_cost += calculate_openai_cost(model, meta_tokens)
                    for key in total_tokens:
                        total_tokens[key] += meta_tokens[key]
                else:
                    answer = llm(metadata_prompt, model=model)
        else:
            # Irrelevant AND not metadata
            print("--> Not a metadata query. Returning graceful failure message.")
            failure_messages = {
                'en': "I could not find a specific answer for your question in the provided documents. Please try rephrasing your query.",
                'es': "No pude encontrar una respuesta específica para tu pregunta en los documentos proporcionados. Por favor, intenta reformular tu consulta.",
                'it': "Non sono riuscito a trovare una risposta specifica alla tua domanda nei documenti forniti. Prova a riformulare la tua domanda."
            }
            answer = failure_messages.get(user_language, failure_messages['en'])
            api_context = []

    # Prepare metrics if evaluation is requested
    if evaluate:
        end_time = time()
        processing_time = end_time - start_time
        
        # Evaluate answer relevance
        evaluation_result, eval_tokens = evaluate_relevance(question, answer)
        total_cost += calculate_openai_cost('gpt-4o-mini', eval_tokens)
        for key in total_tokens:
            total_tokens[key] += eval_tokens[key]
        
        metrics = {
            "processing_time_seconds": round(processing_time, 2),
            "total_cost_usd": round(total_cost, 6),
            "total_tokens": total_tokens,
            "relevance_evaluation": evaluation_result,
            "model_used": model,
            "search_results_count": len(search_results) if search_results else 0
        }
        
        return answer, api_context, metrics
    
    return answer, api_context


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
    
    # Optional '--evaluate' flag for monitoring
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Include evaluation metrics in the response"
    )

    args = parser.parse_args()

    print(f"\n🔍 Query: {args.query} (Language: {args.lang})\n")
    
    if args.evaluate:
        answer, context, metrics = answer_question(args.query, user_language=args.lang, evaluate=True)
        print("\n💡 Answer:\n")
        print(answer)
        print("\n📊 Metrics:\n")
        print(f"Processing Time: {metrics['processing_time_seconds']} seconds")
        print(f"Total Cost: ${metrics['total_cost_usd']}")
        print(f"Total Tokens: {metrics['total_tokens']['total_tokens']}")
        print(f"Relevance: {metrics['relevance_evaluation']['Relevance']}")
        print(f"Evaluation Explanation: {metrics['relevance_evaluation']['Explanation']}")
    else:
        answer, context = answer_question(args.query, user_language=args.lang)
        print("\n💡 Answer:\n")
        print(answer)