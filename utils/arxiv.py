import time
import feedparser
import urllib.request

from rich.panel import Panel
from rich.console import Console

console = Console()

def query_arxiv(search_query, start=0, max_results=50, sort_by="relevance"):
    """
    Query the arXiv API for papers matching the search query.
    
    Args:
        search_query (str): The search query
        start (int): Starting index for results
        max_results (int): Maximum number of results to return
        sort_by (str): Sort order - 'relevance' or 'lastUpdatedDate'
        
    Returns:
        list: List of dictionaries containing paper metadata
        int: Total number of results
    """
    base_url = 'http://export.arxiv.org/api/query?'
    
    # Format the query - Handle both "lastUpdatedDate" and "submittedDate" as valid inputs
    if sort_by in ["lastUpdatedDate", "submittedDate"]:
        sort_criteria = "submittedDate"  # arXiv API uses "submittedDate" for date-based sorting
    else:
        sort_criteria = "relevance"  # Default to relevance
        
    query = f'search_query={search_query}&start={start}&max_results={max_results}&sortBy={sort_criteria}&sortOrder=descending'
    
    # Print the full URL with better formatting
    console_url = base_url + query

    console.print("\n")  # Add spacing before error
    console.print(Panel(
        f"[yellow]{console_url}[/yellow]",
        title=f"[bold white]API Query Endpoint[/bold white]",
        border_style="yellow"
    ))

    # Perform the request
    response = urllib.request.urlopen(base_url + query).read()

    # Parse the response using feedparser
    feed = feedparser.parse(response)

    # Extract paper information
    papers = []

    for entry in feed.entries:
        # Get PDF link
        pdf_link = None
        for link in entry.links:
            if getattr(link, 'title', None) == 'pdf':
                pdf_link = link.href
                break

        # Get all authors
        authors = [author.name for author in entry.authors] if hasattr(entry, 'authors') else [entry.author] if hasattr(entry, 'author') else []

        # Get categories
        categories = [t['term'] for t in entry.tags] if hasattr(entry, 'tags') else []

        # Get journal reference
        journal_ref = getattr(entry, 'arxiv_journal_ref', None)

        # Get comment
        comment = getattr(entry, 'arxiv_comment', None)

        paper = {
            'id': pdf_link.rsplit('/', 1)[-1].split('v', 1)[0],
            'title': entry.title,
            'authors': authors,
            'abstract': entry.summary,
            'published': entry.published,
            'updated': entry.updated,
            'pdf_link': pdf_link,
            'categories': categories,
            'primary_category': categories[0] if categories else None,
            'journal_ref': journal_ref,
            'comment': comment
        }
        papers.append(paper)

    time.sleep(3)   # Be respectful of the API rate limits

    total_results = int(feed.feed.opensearch_totalresults) if hasattr(feed.feed, 'opensearch_totalresults') else 0
    return papers, total_results
