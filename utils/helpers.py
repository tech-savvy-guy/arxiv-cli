import re
import requests
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

# Create a console instance for rich output
console = Console()

def parse_query_with_llm(user_query, client: OpenAI):
    """
    Use an LLM to parse a natural language query into optimized arXiv API parameters.
    
    Args:
        user_query (str): Natural language query from user
        llm_api_key (str): API key for the LLM service
        
    Returns:
        dict: Dictionary containing search parameters
    """

    prompt = f"""
        You are a search query generation assistant for the arXiv API. Your task is to convert natural language input into an arXiv search query string using the proper syntax, field prefixes, Boolean operators, and date range filters.

        Key Points to Consider:

        1. **Field Prefixes:**
        - ti: Title
        - au: Author
        - abs: Abstract
        - co: Comment
        - jr: Journal Reference
        - cat: Subject Category
        - rn: Report Number
        - id: ID (use id_list for multiple IDs)
        - all: All fields

        2. **Boolean Operators:**
        - AND: Both conditions must be met.
        - OR: Either condition can be met.
        - ANDNOT: The first condition must be met, and the second must not be met.

        3. **Date Filter:**
        Format: submittedDate:[YYYYMMDDHHMM+TO+YYYYMMDDHHMM]
        Example: submittedDate:[202301010000+TO+202401010000]

        4. **Grouping:**
        Use parentheses `%28` and `%29` to group Boolean expressions.
        Example: au:"John Doe"+AND+ti:%28quantum+OR+computing%29

        5. **Phrase Searching:**
        Use double quotes `%22` to search for exact phrases within a field.
        Example: ti:%22quantum+computing%22

        6. **URL Encoding:**
        - Replace spaces with `+`
        - Encode special characters: quotes as `%22`, parentheses as `%28` and `%29`

        Your response must be a JSON object with the following structure:
        {{
        "search_query": "<final arXiv query string>",
        "max_results": <number between 5 and 100>,
        "sort_by": "relevance" or "lastUpdatedDate",
        "explanation": "<brief explanation of how the query was constructed>"
        }}

        Do not include any other text.

        User query: "{user_query}"
    """


    completion = client.chat.completions.create(
        model='gpt-4o',
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=500,
    )

    try:
        result = completion.choices[0].message.content
        # Extract JSON from completion
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            import json
            return json.loads(json_match.group(0))
        else:
            # Fallback to simple parsing
            console.print("[bold yellow]Warning:[/bold yellow] Could not parse LLM response properly. Using fallback query.")
            return {
                "search_query": f'all:"{user_query}"',
                "max_results": 30,
                "sort_by": "relevance",
                "explanation": "Fallback to simple all-fields search"
            }
    except Exception as e:
        console.print(f"[bold red]Error parsing LLM response:[/bold red] [yellow]{str(e)}[/yellow]")
        return {
            "search_query": f'all:"{user_query}"',
            "max_results": 30,
            "sort_by": "relevance",
            "explanation": "Error in LLM parsing"
        }

def get_citation_data(arxiv_id: str) -> dict:
    """
    Get citation data for an arXiv paper from Semantic Scholar.

    Args:
        arxiv_id (str): The arXiv ID of the paper

    Returns:
        dict: Citation data including count and influential citations
    """

    url = (
        f"https://api.semanticscholar.org/v1/paper/arXiv:{arxiv_id}"
        "?fields=title,citationCount,influentialCitationCount,references,year"
    )

    default_result = {
        'citation_count': 0,
        'influential_citation_count': 0,
        'references': 0,
        'year': None,
    }

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        return {
            'citation_count': len(data.get('citations', [])),
            'influential_citation_count': data.get('influentialCitationCount', 0),
            'references': len(data.get('references', [])),
            'year': data.get('year'),
        }
    except requests.RequestException as e:
        # Create a better formatted error panel
        error_message = f"Error fetching citation data for {arxiv_id}:\n{str(e)}"
        console.print("\n")  # Add spacing before error
        console.print(Panel(
            f"[yellow]{str(e)}[/yellow]",
            title=f"[bold red]Citation Data Error: {arxiv_id}[/bold red]",
            border_style="red"
        ))
        console.print("\n")  # Add spacing after error
        return default_result
