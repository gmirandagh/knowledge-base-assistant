# Knowledge Base Assistant

A Retrieval-Augmented Generation (RAG) application that transforms your internal technical documents into an interactive and intelligent Q&A system.

<p align="center">
  <img src="images/readme_kba-banner.png" alt="Knowledge Base Assistant Banner">
</p>

## The Problem

Engineers, researchers, and subject matter experts often face the challenge of navigating dense and lengthy technical documents. Finding specific information is time-consuming, and critical knowledge remains siloed within complex PDFs, hindering productivity and knowledge sharing.

## The Solution

The Knowledge Base Assistant provides a powerful conversational AI that ingests these documents and allows users to ask questions in natural language. It makes complex information instantly accessible and actionable, turning your static knowledge base into a dynamic conversation partner.

This project was developed as a Final Master's Thesis for the *Master in Data Science, Big Data & Business Analytics* (UCM-ntic, 2024-2025).

## Key Features

✨ **Interactive Web UI:** A clean, modern web interface for asking questions and receiving answers, complete with an integrated real-time monitoring dashboard.
<br>
📊 **Embedded Grafana Dashboard:** Live, interactive visualizations of system performance, cost, and user feedback are embedded directly into the web UI.
<br>
🤖 **Interactive CLI:** A powerful command-line interface (`cli.py`) for interacting with the assistant, submitting feedback, and viewing system health metrics directly in your terminal.
<br>
🌐 **Multi-Language Support:** The RAG pipeline is capable of handling queries and generating answers in multiple languages (English, Spanish, and Italian supported).
<br>
📈 **Comprehensive Monitoring:** Detailed logging of conversations, feedback, and performance metrics to a PostgreSQL database, visualized in Grafana.
<br>
🐳 **Containerized with Docker:** The entire application stack is containerized for easy, one-command setup and consistent deployment.

## Live Demo & Screenshots

Below is a screenshot of the main web interface, showing the chat application alongside the embedded, real-time Grafana monitoring dashboard.

<p align="center">
  <img src="images/image.png" alt="Web UI with Embedded Grafana Dashboard">
</p>

Watch a full video demo of the project, including setup and usage instructions:

<p align="center">
  <a href="https://www.youtube.com/watch?v=RiQcSHzR8_E">
    <img src="https://markdown-videos-api.jorgenkh.no/youtube/RiQcSHzR8_E" alt="Video Demo">
  </a>
</p>

---

### **Action Items & Questions for You:**

2.  **Update Screenshot:** The current `image.png` might not show the final UI with the embedded dashboard. **Could you please take a new screenshot of the full web page** (chat on top, dashboard below) and replace the existing `images/image.png`? This will be a huge visual upgrade.
3.  **Confirm Master's Program Name:** I kept the name you had. Is `(Online) Máster Data Science, Big Data & Business Analytics 2024-2025 (clase 4)]()` the full, correct title? I've updated it slightly for clarity.

Once you confirm you're happy with this section and have updated the screenshot, we will move on to the next section: **"Tech Stack"** and a radically simplified **"Getting Started"** guide.













-->

## Project overview

The Knowledge Base Assistant is a RAG (Retrieval-Augmented Generation) application designed to serve as an intelligent interface for a library of technical documents.

The main use cases include:

1.  **Specific Information Retrieval:** Answering direct questions by finding the exact relevant text within the document library.
2.  **Conceptual Search:** Recommending related topics or concepts when a user's query is broad.
3.  **Summarization and Synthesis:** Providing summaries of document sections or synthesizing information from multiple sources to answer complex questions.
4.  **Conversational Interaction:** Making it easy to get information without manually sifting through hundreds of pages of documentation.

## Dataset

The dataset used in this project is a collection of technical papers in PDF format. The ETL (Extract, Transform, Load) pipeline processes these documents and converts them into a structured JSONL format, where each record represents a "chunk" of information.

Each chunk record contains:

-   **Document Metadata:** Information about the entire source document, including:
    -   `title`: The title of the paper.
    -   `authors`: A list of the paper's authors.
    -   `year`: The publication year.
    -   `abstract`: A summary of the paper's abstract.
-   **Chunk Data:** Information specific to that piece of text:
    -   `chunk_id`: A unique identifier for the chunk.
    -   `section_title`: The section of the paper the chunk belongs to.
    -   `page_number`: The page number where the chunk originated.
    -   `content`: The raw text of the chunk.
-   **Embedding:** A vector representation of the content for semantic search capabilities.

The final, processed dataset serves as the knowledge base for the assistant.

You can find the data stored in JSONL format in [`data/data.jsonl`](data/data.jsonl).

## Technologies

- Python 3.12
- Docker and Docker Compose for containerization
- [Minsearch](https://github.com/alexeygrigorev/minsearch) for full-text search
- Flask as the API interface (see [Background](#background) for more information on Flask)
- Grafana for monitoring and PostgreSQL as the backend for it
- OpenAI as an LLM

## Preparation

Since we use OpenAI, you need to provide the API key:

1. Install `direnv`. If you use Ubuntu, run `sudo apt install direnv` and then `direnv hook bash >> ~/.bashrc`.
    ```bash
    sudo apt-get update && sudo apt-get install direnv -y
    
    direnv hook bash >> ~/.bashrc
    ```

2. Copy `.envrc_template` into `.envrc` and insert your key there.

3. For OpenAI, it's recommended to create a new project and use a separate key.
    Modify [.env](.env) replacing 'YOUR_KEY' by your actual generated API key.

4. Run `direnv allow` to load the key into your environment.
    ```bash
    direnv allow
    ```

For dependency management, we use pipenv, so you need to install it:

```bash
pip install pipenv
```

Once installed, you can install the app dependencies:

```bash
pipenv install --dev
```

## Running the application

### Database configuration

Before the application starts for the first time, the database
needs to be initialized.

First, run `postgres`:

```bash
docker-compose up postgres
```

Then run the [`db_prep.py`](knowledge_base_assistant/db_prep.py) script:

```bash
pipenv shell

cd knowledge_base_assistant

export POSTGRES_HOST=localhost

python db_prep.py
```

To check the content of the database, use `pgcli` (already
installed with pipenv):

```bash
pipenv run pgcli -h localhost -U your_username -d course_assistant -W
```

You can view the schema using the `\d` command:

```sql
\d conversations;
```

And select from this table:

```sql
select * from conversations;
```

### Running with Docker-Compose

The easiest way to run the application is with `docker-compose`:

```bash
docker-compose up
```

Some quick references for Docker-compose commands to stop and resume in [Background](#background)


### Running locally

If you want to run the application locally,
start only postres and grafana:

```bash
docker-compose up postgres grafana
```

If you previously started all applications with
`docker-compose up`, you need to stop the `app`:

```bash
docker-compose stop app
```

Now run the app on your host machine:

```bash
pipenv shell

cd knowledge_base_assistant

export POSTGRES_HOST=localhost
python app.py
```

### Restarting and cleaning all data

To clean all data from the database and grafana you need to
re-build all containers and database.

To do this, run:

```bash
pipenv shell

docker-compose down -v

docker-compose up --build -d

pipenv run python -m knowledge_base_assistant.db_prep
```



### Running with Docker (without compose)

Sometimes you might want to run the application in
Docker without Docker Compose, e.g., for debugging purposes.

First, prepare the environment by running Docker Compose
as in the previous section.

Next, build the image:

```bash
docker build -t knowledge-base-assistant .
```

And run it:

```bash
docker run -it --rm \
    --network="knowledge-base-assistant_default" \
    --env-file=".env" \
    -p 8000:5000 \
    knowledge-base-assistant
```

### Time configuration

When inserting logs into the database, ensure the timestamps are
correct. Otherwise, they won't be displayed accurately in Grafana.

When you start the application, you will see the following in
your logs:

```
Database timezone: Etc/UTC
Database current time (UTC): 2024-08-24 06:43:12.169624+00:00
Database current time (Europe/Berlin): 2024-08-24 08:43:12.169624+02:00
Python current time: 2024-08-24 08:43:12.170246+02:00
Inserted time (UTC): 2024-08-24 06:43:12.170246+00:00
Inserted time (Europe/Berlin): 2024-08-24 08:43:12.170246+02:00
Selected time (UTC): 2024-08-24 06:43:12.170246+00:00
Selected time (Europe/Berlin): 2024-08-24 08:43:12.170246+02:00
```

Make sure the time is correct.

You can change the timezone by replacing `TZ` in `.env`.

On some systems, specifically WSL, the clock in Docker may get
out of sync with the host system. You can check that by running:

```bash
docker run ubuntu date
```

If the time doesn't match yours, you need to sync the clock:

```bash
wsl

sudo apt install ntpdate
sudo ntpdate time.windows.com
```

Note that the time is in UTC.

After that, start the application (and the database) again.


## Using the application

When the application is running, you can start using it.


### CLI


### Using the RAG CLI

You can query the RAG pipeline directly from your terminal. This is useful for quick tests and scripting.

**Prerequisites:** *Ensure you have completed the setup steps and activated the virtual environment before running these commands.*

#### Usage

```bash
# First, activate the environment
pipenv shell

# Then, run the script
python knowledge_base_assistant/rag.py [options] "Your question goes here"
```

#### Arguments and Options

*   **`query`** (Required): The question you want to ask. It must be the last argument and should be enclosed in quotes.
*   **`--lang {en,es,it}`** (Optional): Specifies the language of the query.
    *   `en`: English (default)
    *   `es`: Spanish
    *   `it`: Italian
*   **`-h, --help`**: Shows the help message with all available options.

---
#### Basic Examples

*   **English Query (Default):**
    ```bash
    python knowledge_base_assistant/rag.py "What is the main conclusion of the document about AI?"
    ```

*   **Spanish Query:**
    ```bash
    python knowledge_base_assistant/rag.py --lang es "¿Cuándo se publicaron los artículos?"
    ```

*   **Italian Query:**
    ```bash
    python knowledge_base_assistant/rag.py --lang it "Chi sono gli autori del documento sull'IA?"
    ```

---
#### Detailed Examples with Output

##### **Example 1: Content Query**

*Command:*
```bash
python knowledge_base_assistant/rag.py "What is the main conclusion of the document about AI?"
```

*Expected Output:*
```bash
✅ Loaded existing index from '/workspaces/knowledge-base-assistant/data/data_index.bin'

🔍 Query: What is the main conclusion of the document about AI? (Language: en)

--> Performing standard content search.

💡 Answer:

The main conclusion of the document emphasizes that the methodologies implemented have led to significant improvements in human performance within organizations. The success of these improvements will depend on tailored implementation strategies and the application of coaching methods. Additionally, the document highlights the importance of measuring performance outcomes through statistical models, which is uncommon in human safety and engineering programs.
```

---
##### **Example 2: Metadata Query**

*Command:*
```bash
python knowledge_base_assistant/rag.py --lang es "¿Cuándo se publicaron los artículos?"
```

*Expected Output:*
```bash
✅ Loaded existing index from '/workspaces/knowledge-base-assistant/data/data_index.bin'

🔍 Query: ¿Cuándo se publicaron los artículos? (Language: es)

--> Original Query (es): ¿Cuándo se publicaron los artículos?
--> Translated English Query: When were the articles published?
--> Detected a metadata query. Retrieving metadata.

💡 Answer:

Los artículos se publicaron en los años 2023 y 2024.
```


### Using the application CLI

We built an interactive CLI application using
[questionary](https://questionary.readthedocs.io/en/stable/).

To start it, run:

```bash
pipenv run python cli.py
```

You can also make it randomly select a question from
[our ground truth dataset](data/ground-truth-retrieval.csv):

```bash
pipenv run python cli.py --random
```

### Using `requests`

When the application is running, you can use
[requests](https://requests.readthedocs.io/en/latest/)
to send questions—use [test.py](test.py) for testing it:

```bash
pipenv run python test.py
```

It will pick a random question from the ground truth dataset
and send it to the app.


### Interacting with the API via cURL

You can interact directly with the Flask API using `curl` or any other API client. The API is served at `http://localhost:5000` when running the application locally.

#### 1. Asking a Question

To ask a question, send a POST request to the `/ask` endpoint with your question in the JSON body.

```bash
# Define variables
URL="http://localhost:5000"
QUESTION="What are the elements of Situational Awareness?"
DATA='{
    "question": "'"${QUESTION}"'"
}'

# Send the request
curl -X POST \
    -H "Content-Type: application/json" \
    -d "${DATA}" \
    "${URL}/ask"
```

The API will respond with the answer, a unique conversation ID, and the sources used for the context:

```json
{
    "question": "What is the main conclusion of the document about AI?",
    "answer": "The main conclusion of the document emphasizes that the methodologies implemented have led to significant improvements in human performance within organizations. The success of these improvements will depend on tailored implementation strategies and the application of coaching methods.",
    "context": [
        {
            "content": "The document highlights the importance of measuring performance outcomes through statistical models, which is uncommon in human safety and engineering programs...",
            "document_id": "doc_1",
            "page_number": "5",
            "section_title": "Conclusion",
            "title": "Improving Human Performance in Organizations",
            "type": "content"
        }
    ],
    "conversation_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef"
}
```

#### 2. Submitting Feedback

To submit feedback for an answer you received, send a POST request to the `/feedback` endpoint. You'll need the `conversation_id` from the previous response.

```bash
# Define variables
URL="http://localhost:5000"
CONVERSATION_ID="a1b2c3d4-e5f6-7890-1234-567890abcdef"
FEEDBACK_DATA='{
    "conversation_id": "'"${CONVERSATION_ID}"'",
    "feedback": 1
}'

# Send the request
curl -X POST \
    -H "Content-Type: application/json" \
    -d "${FEEDBACK_DATA}" \
    "${URL}/feedback"
```

After submitting, you will receive an acknowledgment:

```json
{
    "conversation_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
    "feedback": 1,
    "status": "feedback received"
}
```


## Code

The code for the application is in the [`knowledge_base_assistant`](knowledge_base_assistant/) folder:

- [`app.py`](knowledge_base_assistant/app.py) - the Flask API, the main entrypoint to the application
- [`rag.py`](knowledge_base_assistant/rag.py) - the main RAG logic for building the retrieving the data and building the prompt
- [`ingest.py`](knowledge_base_assistant/ingest.py) - loading the data into the knowledge base
- [`minsearch.py`](knowledge_base_assistant/minsearch.py) - an in-memory search engine
- [`db.py`](knowledge_base_assistant/db.py) - the logic for logging the requests and responses to postgres
- [`db_prep.py`](knowledge_base_assistant/db_prep.py) - the script for initializing the database

We also have some code in the project root directory:

- [`test.py`](test.py) - select a random question for testing
- [`cli.py`](cli.py) - interactive CLI for the APP

### Interface

Flask is used for serving the application as an API.

Refer to the ["Using the Application" section](#using-the-application)
for examples on how to interact with the application.


<!-- TO BE UPDATED -->


### Ingestion

The ingestion script is in [`ingest.py`](knowledge_base_assistant/ingest.py).

Since an in-memory database is used, `minsearch`, as the
knowledge base, the ingestion script is run at the startup
of the application.

It's executed inside [`rag.py`](knowledge_base_assistant/rag.py)
when imported.

## Experiments

For experiments, we use Jupyter notebooks.
They are in the [`notebooks`](notebooks/) folder.

To start Jupyter, run:

```bash
cd notebooks
pipenv run jupyter notebook
```

We have the following notebooks:

- [`rag-test.ipynb`](notebooks/rag-test.ipynb): The RAG flow and evaluating the system.
- [`evaluation-data-generation.ipynb`](notebooks/evaluation-data-generation.ipynb): Generating the ground truth dataset for retrieval evaluation.

### Retrieval evaluation

The basic approach - using `minsearch` without any boosting - gave the following metrics:

- Hit rate: 90%
- MRR: 70%

### RAG flow evaluation

We used the LLM-as-a-Judge metric to evaluate the quality
of our RAG flow.

For `gpt-4o-mini`, in a sample with 200 records, we had:

- 155 (78%) `RELEVANT`
- 16 (8%) `PARTLY_RELEVANT`
- 29 (14.5%) `NON_RELEVANT`

## Monitoring

We use Grafana for monitoring the application. 

It's accessible at [localhost:3000](http://localhost:3000):

- Login: "admin"
- Password: "admin"

### Dashboards

<p align="center">
  <img src="images/dash.png">
</p>

The monitoring dashboard contains several panels:

1. **Last 5 Conversations (Table):** Displays a table showing the five most recent conversations, including details such as the question, answer, relevance, and timestamp. This panel helps monitor recent interactions with users.
2. **+1/-1 (Pie Chart):** A pie chart that visualizes the feedback from users, showing the count of positive (thumbs up) and negative (thumbs down) feedback received. This panel helps track user satisfaction.
3. **Relevancy (Gauge):** A gauge chart representing the relevance of the responses provided during conversations. The chart categorizes relevance and indicates thresholds using different colors to highlight varying levels of response quality.
4. **OpenAI Cost (Time Series):** A time series line chart depicting the cost associated with OpenAI usage over time. This panel helps monitor and analyze the expenditure linked to the AI model's usage.
5. **Tokens (Time Series):** Another time series chart that tracks the number of tokens used in conversations over time. This helps to understand the usage patterns and the volume of data processed.
6. **Model Used (Bar Chart):** A bar chart displaying the count of conversations based on the different models used. This panel provides insights into which AI models are most frequently used.
7. **Response Time (Time Series):** A time series chart showing the response time of conversations over time. This panel is useful for identifying performance issues and ensuring the system's responsiveness.

### Setting up Grafana

All Grafana configurations are in the [`grafana`](grafana/) folder:

- [`init.py`](grafana/init.py) - for initializing the datasource and the dashboard.
- [`dashboard.json`](grafana/dashboard.json) - the actual dashboard (taken from LLM Zoomcamp without changes).

To initialize the dashboard, first ensure Grafana is
running (it starts automatically when you do `docker-compose up`).

Then run:

```bash
pipenv shell

cd grafana

# make sure the POSTGRES_HOST variable is not overwritten 
env | grep POSTGRES_HOST

python init.py
```

Then go to [localhost:3000](http://localhost:3000):

- Login: "admin"
- Password: "admin"

When prompted, keep "admin" as the new password.

## Background

This section is to provide background on some tech not used in the Master and links for further reading.

### Flask

Flask is used for creating the API interface for the application.
It's a web application framework for Python: an endpoint can easily 
be created  for asking questions and use web clients
(like `curl` or `requests`) for communicating with it.

In this case, questions can be sent to `http://localhost:5000/question`.

For more information, visit the [official Flask documentation](https://flask.palletsprojects.com/).


### Docker-compose quick reference

A few useful commands to stop and resume the docker server (and dependencies):
- *Stop everything*: docker-compose down
- *Start everything in background*: docker-compose up -d
- *View status*: docker-compose ps
- *View logs*: docker-compose logs -f app

# Stop + remove volumes (⚠️ deletes all data!)
docker-compose down -v


## Acknowledgements 

I wish to thank the UCM-ntic staff for the Master in Data Science, Big Data & Business Analytics (2024-2025).
Your exceptional organization and insightful content provided a vital foundation and ignited a passion for 
further exploration into Data Science, AI, and state-of-the-art technologies.