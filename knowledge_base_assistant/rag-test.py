#!/usr/bin/env python
# coding: utf-8

# ## Ingestion

# In[1]:


import os
import minsearch
import json
import pickle


PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..')) 

data_path = os.path.join(PROJECT_ROOT, 'data', 'data.jsonl')
data_index_path = os.path.join(PROJECT_ROOT, 'data', 'data_index.bin')

# Load processed data
with open(data_path, 'rt', encoding='utf-8') as f_in:
    docs = [json.loads(line) for line in f_in]

# Filter out any chunks where embedding failed (is None)
documents = [doc for doc in docs if doc.get('embedding') is not None]
print(f"Loaded {len(documents)} documents with embeddings.")

# Create Minsearch Index
index = minsearch.Index(
    text_fields=["text"],
    keyword_fields=["title", "document_id", "chunk_type", "section_title"]
)

# Add documents to the index
index.fit(documents)

# SAVE THE INDEX
with open(data_index_path, 'wb') as f_out:
    pickle.dump(index, f_out)

print(f"\nMinsearch KEYWORD index created and saved successfully to '{data_index_path}'")


# ## RAG flow

# In[2]:


import minsearch
import json
import os
import pickle

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..')) 
data_index_path = os.path.join(PROJECT_ROOT, 'data', 'data_index.bin')

# Load the Index
with open(data_index_path, 'rb') as f_in:
    index = pickle.load(f_in)

print("Minsearch index loaded successfully.")


# In[3]:


from openai import OpenAI
client = OpenAI()


# In[4]:


def search(query, filter_dict={}):
    """Performs a keyword search with optional filtering."""
    boost = {'title': 3.0, 'section_title': 1.5}
    # boost = {}
    results = index.search(
        query=query,
        filter_dict=filter_dict,
        boost_dict=boost,
        num_results=7
    )
    return results


# In[5]:


# Main prompt to be sent to the LLM
prompt_template = """
You are an expert assistant for a technical knowledge base.
Your task is to answer the user's QUESTION based *only* on the provided CONTEXT from the document library.
If the CONTEXT does not contain the answer, state that the information is not available in the provided documents.
Be concise, accurate, and cite the source document title when possible.

QUESTION:
{question}

CONTEXT:
---
{context}
---
""".strip()

# Single retrieved chunk for the LLM labeling the source and context for the model
entry_template = """
Source Document: {title}
Section: {section_title} (Page {page_number})
Content: {content}
""".strip()

# Assembles final prompt for the LLM
def build_prompt(query, search_results):
    context_parts = []

    for doc in search_results:
        context_parts.append(entry_template.format(**doc))

    context = "\n\n---\n\n".join(context_parts)

    prompt = prompt_template.format(question=query, context=context)
    return prompt


# In[6]:


def llm(prompt, model='gpt-4o-mini'):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# In[68]:


def rag(query, model='gpt-4o-mini'):
    """
    Advanced, hybrid RAG function that:
    1. Checks for metadata queries.
    2. If found, retrieves all relevant metadata.
    3. Passes the metadata and the original query to an LLM for reasoning.
    4. If not a metadata query, performs standard content search and synthesis.
    """
    import os

    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))    
    data_path = os.path.join(PROJECT_ROOT, 'data', 'data.jsonl')

    # Step 1: Detect if it's a Metadata Query
    metadata_keywords = ['author', 'authors', 'year', 'published', 'title', 'document id']
    is_metadata_query = any(keyword in query.lower() for keyword in metadata_keywords)

    if is_metadata_query:
        print("--> Detected a metadata query. Retrieving all metadata for context.")

        # Load dataset
        with open(data_path, 'rt', encoding='utf-8') as f_in:
            all_docs = [json.loads(line) for line in f_in]

        # Get metadata
        papers_metadata = {}
        for doc in all_docs:
            title = doc['document_metadata'].get('title')
            if title and title not in papers_metadata:
                papers_metadata[title] = doc['document_metadata']

        #  Step 2: Build Context from Metadata
        context = "Here is the available metadata from the knowledge base:\n\n"
        for i, (title, meta) in enumerate(papers_metadata.items()):
            context += f"--- Document {i+1} ---\n"
            context += f"Title: {meta.get('title', 'N/A')}\n"
            context += f"Authors: {', '.join(meta.get('authors', []))}\n"
            context += f"Year: {meta.get('year', 'N/A')}\n\n"

        # Step 3: Build Prompt and Call the LLM for Reasoning
        metadata_prompt = f"""
You are a research assistant. Your task is to answer the user's QUESTION based *only* on the provided METADATA.
Perform any necessary analysis, such as comparing lists or counting, to answer the question accurately.

QUESTION:
{query}

METADATA:
---
{context}
---
""".strip()

        # Call the LLM to reason over metadata
        answer = llm(metadata_prompt, model=model)
        return answer

    else:
        print("--> Performing standard content search.")
        search_results = search(query)
        if not search_results:
            return "I could not find any relevant information in the documents for that query."

        prompt = build_prompt(query, search_results)
        answer = llm(prompt, model=model)
        return answer


# In[8]:


question = "What does CRM tand for?"


# In[9]:


answer = rag(question)
print(answer)


# ## Retrieval evaluation

# In[10]:


import os
import json
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..')) 
csv_output_path = os.path.join(PROJECT_ROOT, 'data', 'ground-truth-retrieval.csv')

df_question = pd.read_csv(csv_output_path)
df_question.head()


# In[11]:


ground_truth = df_question.to_dict(orient='records')


# In[12]:


ground_truth[0]


# In[13]:


def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)

def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)


# In[14]:


def minsearch_search(query, filter_dict={}):
    # boost = {
    #     "title": 2.0,
    #     "section_title": 1.5,
    #     "document_metadata": 1.2,
    #     "content": 0.9,
    # }

    boost = {}
    results = index.search(
        query=query,
        filter_dict=filter_dict,
        boost_dict=boost,
        num_results=12
    )

    return results


# In[15]:


def evaluate(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        doc_id = q['id']
        results = search_function(q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }


# In[16]:


from tqdm.auto import tqdm


# In[17]:


evaluate(ground_truth, lambda q: minsearch_search(q['question']))


# In[18]:


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# Score-enabled search
# ------------------------------
def minsearch_search_with_scores(query, filter_dict={}, boost_dict={}, num_results=12):
    # Build query vectors for each text field
    query_vecs = {
        field: index.vectorizers[field].transform([query])
        for field in index.text_fields
    }

    scores = np.zeros(len(index.docs))

    # Apply cosine similarity + boosts
    for field, query_vec in query_vecs.items():
        sim = cosine_similarity(query_vec, index.text_matrices[field]).flatten()
        boost = boost_dict.get(field, 1.0)
        scores += sim * boost

    # Apply filters
    for field, value in filter_dict.items():
        if field in index.keyword_fields:
            mask = (index.keyword_df[field] == value).to_numpy()
            scores = scores * mask

    # Top-k
    top_indices = np.argpartition(scores, -num_results)[-num_results:]
    top_indices = top_indices[np.argsort(-scores[top_indices])]

    results = [(index.docs[i], float(scores[i])) for i in top_indices if scores[i] > 0]

    return results


# ------------------------------
# Extra metrics
# ------------------------------
def average_precision(relevance):
    """AP for one query"""
    hits, sum_prec = 0, 0.0
    for i, rel in enumerate(relevance, start=1):
        if rel:
            hits += 1
            sum_prec += hits / i
    return sum_prec / hits if hits > 0 else 0.0

def ndcg(relevance, k=None):
    """Normalized Discounted Cumulative Gain"""
    if k is None:
        k = len(relevance)
    rel = np.array(relevance[:k], dtype=int)
    dcg = np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))
    ideal = np.sum(sorted(rel, reverse=True) / np.log2(np.arange(2, len(rel) + 2)))
    return dcg / ideal if ideal > 0 else 0.0


# ------------------------------
# Evaluation with scores
# ------------------------------
def evaluate_with_scores(ground_truth, search_function, k=12):
    relevance_total = []
    ap_scores = []
    ndcg_scores = []
    relevant_doc_scores = []

    for q in tqdm(ground_truth):
        doc_id = q['id']
        results = search_function(q['question'], num_results=k)

        # Extract docs and scores
        docs, scores = zip(*results) if results else ([], [])
        relevance = [d['id'] == doc_id for d in docs]

        # Store metrics
        relevance_total.append(relevance)
        ap_scores.append(average_precision(relevance))
        ndcg_scores.append(ndcg(relevance, k=k))

        # If relevant doc was retrieved, log its score
        for d, s in results:
            if d['id'] == doc_id:
                relevant_doc_scores.append(s)

    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
        'map': np.mean(ap_scores),
        'ndcg': np.mean(ndcg_scores),
        'avg_relevant_score': np.mean(relevant_doc_scores) if relevant_doc_scores else 0.0,
    }


# ------------------------------
# Run evaluation
# ------------------------------
metrics = evaluate_with_scores(
    ground_truth,
    lambda q, num_results=12: minsearch_search_with_scores(q, num_results=num_results)
)

print(metrics)


# In[46]:


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

# ---------------------------------------
# Score-enabled search (same as before)
# ---------------------------------------
def minsearch_search_with_scores(query, filter_dict={}, boost_dict={}, num_results=12):
    query_vecs = {
        field: index.vectorizers[field].transform([query])
        for field in index.text_fields
    }

    scores = np.zeros(len(index.docs))

    for field, query_vec in query_vecs.items():
        sim = cosine_similarity(query_vec, index.text_matrices[field]).flatten()
        boost = boost_dict.get(field, 1.0)
        boost = boost
        scores += sim * boost

    for field, value in filter_dict.items():
        if field in index.keyword_fields:
            mask = (index.keyword_df[field] == value).to_numpy()
            scores = scores * mask

    top_indices = np.argpartition(scores, -num_results)[-num_results:]
    top_indices = top_indices[np.argsort(-scores[top_indices])]

    results = [(index.docs[i], float(scores[i])) for i in top_indices if scores[i] > 0]

    return results


# ---------------------------------------
# Extra metrics
# ---------------------------------------
def average_precision(relevance):
    hits, sum_prec = 0, 0.0
    for i, rel in enumerate(relevance, start=1):
        if rel:
            hits += 1
            sum_prec += hits / i
    return sum_prec / hits if hits > 0 else 0.0

def ndcg(relevance, k=None):
    if k is None:
        k = len(relevance)
    rel = np.array(relevance[:k], dtype=int)
    dcg = np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))
    ideal = np.sum(sorted(rel, reverse=True) / np.log2(np.arange(2, len(rel) + 2)))
    return dcg / ideal if ideal > 0 else 0.0


# ---------------------------------------
# Evaluation with per-query details
# ---------------------------------------
def evaluate_with_scores(ground_truth, search_function, k=12):
    relevance_total = []
    ap_scores, ndcg_scores, relevant_doc_scores = [], [], []
    per_query_details = []

    for q in tqdm(ground_truth):
        doc_id = q['id']
        query_text = q['question']

        results = search_function(query_text, num_results=k)
        docs, scores = zip(*results) if results else ([], [])
        relevance = [d['id'] == doc_id for d in docs]

        # Compute metrics for this query
        ap = average_precision(relevance)
        ndcg_score = ndcg(relevance, k=k)
        rank = next((i + 1 for i, rel in enumerate(relevance) if rel), None)
        score = next((s for d, s in results if d['id'] == doc_id), None)

        relevance_total.append(relevance)
        ap_scores.append(ap)
        ndcg_scores.append(ndcg_score)
        if score is not None:
            relevant_doc_scores.append(score)

        per_query_details.append({
            'query': query_text,
            'ground_truth_id': doc_id,
            'retrieved_ids': [d['id'] for d in docs],
            'relevance': relevance,
            'rank': rank,                # rank of the correct doc (1-based), or None if missing
            'score': score,              # cosine similarity score for correct doc, or None
            'ap': ap,
            'ndcg': ndcg_score
        })

    # Aggregate
    metrics = {
        'hit_rate': float(hit_rate(relevance_total)),
        'mrr': float(mrr(relevance_total)),
        'map': float(np.mean(ap_scores)),
        'ndcg': float(np.mean(ndcg_scores)),
        'avg_relevant_score': float(np.mean(relevant_doc_scores)) if relevant_doc_scores else 0.0,
    }

    return metrics, per_query_details


# In[47]:


metrics, details = evaluate_with_scores(
    ground_truth,
    lambda q, num_results=12: minsearch_search_with_scores(q, num_results=num_results)
)

print("Aggregated metrics:")
print(metrics)

print("\nExample query breakdown:")
for d in details[:3]:  # show first 3 queries
    print(d)


# ## Finding the best parameters

# In[22]:


df_validation = df_question[:100]
df_test = df_question[100:]


# In[23]:


import random

def simple_optimize(param_ranges, objective_function, n_iterations=10):
    best_params = None
    best_score = float('-inf')  # Assuming we're minimizing. Use float('-inf') if maximizing.

    for _ in range(n_iterations):
        # Generate random parameters
        current_params = {}
        for param, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                current_params[param] = random.randint(min_val, max_val)
            else:
                current_params[param] = random.uniform(min_val, max_val)

        # Evaluate the objective function
        current_score = objective_function(current_params)

        # Update best if current is better
        if current_score > best_score:  # Change to > if maximizing
            best_score = current_score
            best_params = current_params

    return best_params, best_score


# In[24]:


gt_val = df_validation.to_dict(orient='records')


# In[41]:


def minsearch_search(query, boost=None):
    if boost is None:
        boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=12
    )

    return results


# In[42]:


param_ranges = {
    'text': (0.0, 3.0),
    'title': (0.0, 3.0),
    'document_id': (0.0, 3.0),
    'chunk_type': (0.0, 3.0),
    'section_title': (0.0, 3.0),
    'page_number': (0.0, 3.0),
    'content': (0.0, 3.0),
    'id': (0.0, 3.0),
    'url': (0.0, 3.0)
}

def objective(boost_params):
    def search_function(q):
        return minsearch_search(q['question'], boost_params)

    results = evaluate(gt_val, search_function)
    return results['mrr']


# In[43]:


simple_optimize(param_ranges, objective, n_iterations=20)


# ## RAG evaluation

# In[48]:


prompt2_template = """
You are an expert evaluator for a RAG system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


# In[50]:


len(ground_truth)


# In[54]:


record = ground_truth[0]
question = record['question']
answer_llm = rag(question)


# In[55]:


print(answer_llm)


# In[56]:


prompt = prompt2_template.format(question=question, answer_llm=answer_llm)
print(prompt)


# In[64]:


import json


# In[65]:


df_sample = df_question.sample(n=200, random_state=1)


# In[66]:


sample = df_sample.to_dict(orient='records')


# In[69]:


evaluations = []

for record in tqdm(sample):
    question = record['question']
    answer_llm = rag(question) 

    prompt = prompt2_template.format(
        question=question,
        answer_llm=answer_llm
    )

    evaluation = llm(prompt)
    evaluation = json.loads(evaluation)

    evaluations.append((record, answer_llm, evaluation))


# In[70]:


df_eval = pd.DataFrame(evaluations, columns=['record', 'answer', 'evaluation'])

df_eval['id'] = df_eval.record.apply(lambda d: d['id'])
df_eval['question'] = df_eval.record.apply(lambda d: d['question'])

df_eval['relevance'] = df_eval.evaluation.apply(lambda d: d['Relevance'])
df_eval['explanation'] = df_eval.evaluation.apply(lambda d: d['Explanation'])

del df_eval['record']
del df_eval['evaluation']


# In[75]:


df_eval.relevance.value_counts()


# In[72]:


df_eval.relevance.value_counts(normalize=True)


# In[74]:


df_eval[df_eval.relevance == 'NON_RELEVANT']


# In[73]:


df_eval.to_csv('../data/rag-eval-gpt-4o-mini.csv', index=False)


# In[ ]:




