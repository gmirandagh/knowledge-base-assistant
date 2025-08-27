import sys
import os
import requests
import questionary
import argparse
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from questionary import prompt

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
DEFAULT_CSV_FILE = "data/ground-truth-retrieval.csv"

console = Console()

def get_random_question(file_path: str):
    """Picks a random question from the ground truth CSV."""
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            console.print(f"[bold red]Error:[/bold red] The CSV file '{file_path}' is empty.")
            return None
        return df.sample(n=1).iloc[0]["question"]
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Ground truth file not found at '{file_path}'.")
        return None
    except KeyError:
        console.print(f"[bold red]Error:[/bold red] CSV file must have a 'question' column.")
        return None
    except Exception as e:
        console.print(f"[bold red]Error reading CSV:[/bold red] {e}")
        return None

def ask_question(question: str, enable_monitoring: bool):
    """Sends a question to the RAG API."""
    url = f"{BASE_URL}/ask"
    payload = {"question": question}
    if enable_monitoring:
        payload["enable_monitoring"] = True
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        console.print(f"\n[bold red]API Error:[/bold red] Could not connect to the application at {url}.")
        console.print("Please ensure the application is running.")
        return None

def send_feedback(conversation_id: str, feedback_value: int):
    """Sends feedback for a conversation."""
    url = f"{BASE_URL}/feedback"
    payload = {"conversation_id": conversation_id, "feedback": feedback_value}
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.status_code
    except requests.exceptions.RequestException as e:
        console.print(f"\n[bold red]Feedback API Error:[/bold red] {e}")
        return None

def display_metrics(metrics: dict):
    """Displays monitoring metrics for a single request."""
    if not metrics:
        return
    table = Table(title="[bold cyan]Request Metrics[/bold cyan]", show_header=False, box=None)
    table.add_column("Metric", style="dim")
    table.add_column("Value")
    table.add_row("Model Used", metrics.get('model_used', 'N/A'))
    table.add_row("Response Time", f"{metrics.get('processing_time_seconds', 0):.2f}s")
    table.add_row("Total Cost", f"${metrics.get('total_cost_usd', 0):.6f}")
    relevance = metrics.get('relevance_evaluation', {}).get('Relevance', 'N/A')
    table.add_row("Relevance", relevance)
    console.print(table)

def handle_feedback(conversation_id: str):
    """Handles the user feedback prompt using the robust `questionary.prompt` method."""
    if not sys.stdin.isatty():
        return
    questions = [{'type': 'select', 'name': 'feedback_choice', 'message': 'Was this answer helpful?', 'choices': ["👍 +1 (Helpful)", "👎 -1 (Not Helpful)", "Skip Feedback"]}]
    result = prompt(questions)
    if not result:
        return
    feedback_choice = result.get('feedback_choice')
    if feedback_choice and "Skip" not in feedback_choice:
        feedback_value = 1 if "👍" in feedback_choice else -1
        status = send_feedback(conversation_id, feedback_value)
        if status == 200:
            console.print("[green]Feedback sent successfully![/green]")
        else:
            console.print("[red]Failed to send feedback.[/red]")

def display_system_metrics():
    """Fetches and displays key system metrics from the app's API."""
    try:
        conv_response = requests.get(f"{BASE_URL}/stats/conversations?days=7", timeout=10)
        feed_response = requests.get(f"{BASE_URL}/stats/feedback", timeout=10)
        conv_response.raise_for_status()
        feed_response.raise_for_status()
        
        conv_data = conv_response.json()
        feed_data = feed_response.json()

        table = Table(title="[bold yellow]System Health Overview (Last 7 Days)[/bold yellow]")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        table.add_row("Total Conversations", str(conv_data.get('total_conversations', 'N/A')))
        
        avg_time = conv_data.get('avg_response_time', 0)
        table.add_row("Avg. Response Time", f"{avg_time:.2f} s")
        
        total_cost = conv_data.get('total_cost', 0)
        table.add_row("Total Cost (USD)", f"${total_cost:.4f}")

        thumbs_up = feed_data.get('thumbs_up', 0)
        thumbs_down = feed_data.get('thumbs_down', 0)
        score = thumbs_up - thumbs_down
        score_text = f"{'+' if score > 0 else ''}{score} ({thumbs_up} 👍 / {thumbs_down} 👎)"
        table.add_row("Feedback Score", score_text)
        
        console.print(table)

    except requests.exceptions.RequestException as e:
        console.print(f"\n[red]Could not fetch system metrics from API: {e}[/red]")
    except Exception as e:
        console.print(f"\n[red]An error occurred while displaying system metrics: {e}[/red]")


def main():
    parser = argparse.ArgumentParser(description="Interactive CLI for the Knowledge Base Assistant.")
    parser.add_argument("--random", action="store_true", help="Use a random question from the ground truth file.")
    parser.add_argument("--file", default=DEFAULT_CSV_FILE, help=f"Path to the ground truth CSV file.")
    args = parser.parse_args()

    console.print(Panel("[bold green]Welcome to the Knowledge Base Assistant CLI![/bold green]", expand=False))
    console.print("Press Ctrl+C to exit.")

    display_system_metrics()

    while True:
        try:
            question = ""
            if args.random:
                question = get_random_question(args.file)
                if not question: break
                console.print(f"\n[yellow]Random Question:[/yellow] {question}")
            else:
                question = questionary.text("\nEnter your question:").ask()
                if question is None: break
            
            enable_monitoring = questionary.confirm("Enable detailed monitoring for this request?").ask()
            if enable_monitoring is None: break

            response = ask_question(question, enable_monitoring)

            if response:
                console.print(Panel(response.get("answer", "[italic]No answer.[/italic]"), title="[bold]Answer[/bold]", border_style="green"))
                display_metrics(response.get("metrics"))
                conversation_id = response.get("conversation_id")
                if conversation_id:
                    handle_feedback(conversation_id)
            
            console.print("-" * 50)
            
            display_system_metrics()

        except (KeyboardInterrupt, EOFError):
            break

    console.print("\n[bold]Goodbye![/bold]")


if __name__ == "__main__":
    main()