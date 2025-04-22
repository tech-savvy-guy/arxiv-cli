import os
import time
import dotenv
import keyboard
import webbrowser
from openai import OpenAI

import typer
from rich.text import Text
from rich.align import Align
from rich.panel import Panel
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn

from utils.arxiv import query_arxiv
from utils.ranking import rank_papers
from utils.helpers import parse_query_with_llm


app = typer.Typer()
console = Console()
dotenv.load_dotenv()  # Load environment variables


def pause(seconds=1):
    time.sleep(seconds)


def display_paper(paper, index, is_selected=False):
    """Display a single paper in a beautiful non-tabular format"""
    # Enhanced visual feedback for selection
    border_style = "bright_yellow" if is_selected else "bright_blue"
    title_style = "bold yellow on black" if is_selected else "bold cyan"
    
    # Create panel title with index and title
    title = f"[{index}] [{title_style}]{paper['title']}[/{title_style}]"
    
    # Add selection indicator
    # if is_selected:   title = "➤ " + title
    
    # Format authors
    authors = ", ".join(paper['authors'][:5])
    if len(paper['authors']) > 5:
        authors += "..."
    
    # Format citation info if available
    citation_info = ""
    if 'citation_data' in paper:
        cites = paper['citation_data'].get('citation_count', 'N/A')
        citation_info = f"\n[bold yellow]Citations:[/bold yellow] {cites}"
    
    # Format content
    content = f"[bold magenta]Authors:[/bold magenta] {authors}\n"
    content += f"[bold green]Published:[/bold green] {paper['published'][:10]}\n"
    content += f"[bold blue]arXiv ID:[/bold blue] {paper['id']}{citation_info}\n"
    content += f"[bold]PDF:[/bold] {paper['pdf_link']}\n\n"
    
    # Format abstract (first 300 chars with ellipsis)
    abstract = paper['abstract']
    if len(abstract) > 300:
        abstract = abstract[:300] + "..."
    content += f"[italic]{abstract}[/italic]"
    
    # Create and return the panel with different border style if selected
    return Panel(
        content,
        title=title,
        border_style=border_style,
        padding=(1, 2)
    )


def display_papers_page(papers, page):
    """Display a single paper per page with pagination"""
    total_pages = len(papers)
    idx = page  # Using page as direct index (adjusted from 1-based to 0-based later)
    
    # Clear the console before displaying anything
    console.clear()
    console.print(f"\n[bold cyan]Paper {page}/{total_pages}[/bold cyan]\n")
    
    # Display the current paper (always selected)
    paper_idx = page - 1  # Convert to 0-based index
    if 0 <= paper_idx < len(papers):
        console.print(display_paper(papers[paper_idx], page, True))
    
    # Navigation help
    console.print("\n[bold yellow]Navigation:[/bold yellow]")
    console.print("[dim]→/← or Right/Left Arrow: Next/Previous paper")
    console.print("[dim]Enter: Open current paper's PDF")
    console.print("[dim]D: Show detailed view of current paper")
    console.print("[dim]Space: Start a new search")
    console.print("[dim]Q: Quit[/dim]")
    return total_pages


def handle_keyboard_input():
    """Wait for keyboard input with better error handling"""
    try:
        # Wait for a key event
        key_event = keyboard.read_event(suppress=True)
        if key_event.event_type == keyboard.KEY_DOWN:
            return key_event.name.lower()
        return None
    except Exception as e:
        console.print(f"[dim]Keyboard input error: {str(e)}[/dim]")
        return None


def show_paper_details(paper):
    """Show detailed view of a single paper"""
    console.clear()
    
    # Title
    console.print(f"[bold cyan]{paper['title']}[/bold cyan]\n")
    
    # Authors (all of them)
    console.print("[bold magenta]Authors:[/bold magenta]")
    for author in paper['authors']:
        console.print(f"• {author}")
    
    # Metadata
    console.print(f"\n[bold green]Published:[/bold green] {paper['published'][:10]}")
    console.print(f"[bold blue]arXiv ID:[/bold blue] {paper['id']}")
    console.print(f"[bold]URL:[/bold] https://arxiv.org/abs/{paper['id']}")
    console.print(f"[bold]PDF:[/bold] {paper['pdf_link']}")
    
    # Categories
    if 'categories' in paper and paper['categories']:
        console.print("\n[bold yellow]Categories:[/bold yellow]")
        for category in paper['categories']:
            console.print(f"• {category}")
    
    # Citation data if available
    if 'citation_data' in paper:
        cites = paper['citation_data']
        console.print("\n[bold yellow]Citation Data:[/bold yellow]")
        console.print(f"• Citation count: {cites.get('citation_count', 'N/A')}")
        console.print(f"• Influential citations: {cites.get('influential_citation_count', 'N/A')}")
        console.print(f"• References: {cites.get('references', 'N/A')}")
        console.print(f"• Year: {cites.get('year', 'N/A')}")
    
    # Abstract
    console.print("\n[bold]Abstract:[/bold]")
    console.print(Text(paper['abstract'], style="italic"))
    
    # Ranking scores if available
    if 'scores' in paper:
        s = paper['scores']
        console.print("\n[dim italic]Ranking metrics:[/dim italic]")
        console.print(f"[dim]Relevance: {s['relevance']:.2f} | Citations: {s['citation']:.2f} | " +
                     f"Recency: {s['recency']:.2f} | Combined: {s['combined']:.2f}[/dim]")
    
    # Navigation instructions
    console.print("\n[dim]Press Enter to go back to results | O to open PDF[/dim]")
    
    while True:
        key = handle_keyboard_input()
        if key in ('enter', 'esc'):
            return
        elif key == 'o':
            try:
                webbrowser.open(paper['pdf_link'])
                console.print("[dim]Opening PDF in browser...[/dim]")
                pause(1)
            except Exception as e:
                console.print(f"[bold red]Error opening PDF:[/bold red] {str(e)}")
                pause(1)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Main callback to run when no subcommand is used"""
    if ctx.invoked_subcommand is None:
        # If no subcommand is provided, run the start function
        start()


def start():
    console.clear()

    # Intro panel with two lines of text
    intro_content = (
        "[bold cyan]    Welcome to[/bold cyan] [bright_white on black] Smart Research CLI[/bright_white on black]\n"
        "[dim]AI-enhanced exploration of arXiv topics[/dim]"
    )

    intro_panel = Panel(
        intro_content,
        title="🔍 arXiv Assistant",
        subtitle="Press Ctrl+C to exit anytime",
        padding=(1, 6),
        border_style="cyan"
    )
    
    # Print the intro panel with centered content
    console.print(Align.center(intro_panel))

    pause(1.5)

    search_loop = True
    while search_loop:
        # Check for OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            console.print("\n[bold red]Error:[/bold red] OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
            return

        # Initialize OpenAI Client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Prompt user
        console.print("\n")
        topic = Prompt.ask("[green]🧠 What topic are you exploring today[/green]")
        
        use_citations = Confirm.ask("\n[yellow]Include citation data?[/yellow] (slower but improves ranking)", default=True)

        console.print(f"\n[bold green]Great![/bold green] I'll try to find relevant research papers on:\n\n[cyan italic]{topic}[/cyan italic]\n")
        
        # Show spinner while searching
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[bold blue]Searching arXiv...[/bold blue]", total=None)
            
            try:
                # Parse query using LLM
                query_params = parse_query_with_llm(topic, client)
                
                # Query arXiv with the optimized query
                papers, total_results = query_arxiv(
                    query_params['search_query'], 
                    max_results=query_params['max_results'],
                    sort_by=query_params['sort_by']
                )
                
                # Pass the existing progress object to rank_papers to avoid flickering
                ranked_papers = rank_papers(papers, topic, use_citations=use_citations, external_progress=progress)
            except Exception as e:
                console.print(f"\n[bold orange1]Error:[/bold orange1] [yellow]{str(e)}[/yellow]")
                return
        
        if not ranked_papers:
            console.print("\n[bold yellow]No papers found matching your query.[/bold yellow]")
            continue
        
        # Display search metadata
        console.print(f"\n[dim]Found {len(ranked_papers)} papers from arXiv (out of {total_results} total matches)[/dim]")
        console.print(f"[dim]Using query: {query_params['search_query']}[/dim]")
        
        pause(1)
        
        # Pagination variables - now showing one paper per page
        current_page = 1
        
        # Display initial page
        total_pages = display_papers_page(ranked_papers, current_page)
        
        # Flush any pending key events
        try:
            while keyboard.is_pressed("any"):
                keyboard.read_key(suppress=True)
                time.sleep(0.1)
        except:
            pass
        
        # Interactive navigation loop with arrow keys
        navigation_loop = True
        while navigation_loop:
            # Wait for a key press
            key = handle_keyboard_input()
            
            if not key:
                continue
                
            if key == 'q':
                search_loop = False
                navigation_loop = False
                return  # Exit the application
            
            elif key == 'space':
                # Exit the navigation loop to start a new search
                navigation_loop = False
                console.clear()
            
            elif key in ('right', 'n') and current_page < total_pages:
                current_page += 1
                # Display the new page
                display_papers_page(ranked_papers, current_page)
            
            elif key in ('left', 'p') and current_page > 1:
                current_page -= 1
                # Display the new page
                display_papers_page(ranked_papers, current_page)
            
            elif key in ('enter', 'o'):
                # Open the PDF for the current paper
                paper_idx = current_page - 1
                if 0 <= paper_idx < len(ranked_papers):
                    try:
                        pdf_link = ranked_papers[paper_idx]['pdf_link']
                        webbrowser.open(pdf_link)
                        console.print("[dim]Opening PDF in browser...[/dim]")
                        # Redisplay the page after a brief pause
                        pause(1)
                        display_papers_page(ranked_papers, current_page)
                    except Exception as e:
                        console.print(f"[bold red]Error opening PDF:[/bold red] {str(e)}")
                        pause(1)
                        display_papers_page(ranked_papers, current_page)
            
            elif key == 'd':
                # Show details for current paper
                paper_idx = current_page - 1
                if 0 <= paper_idx < len(ranked_papers):
                    show_paper_details(ranked_papers[paper_idx])
                    # Redisplay the page after viewing details
                    display_papers_page(ranked_papers, current_page)


if __name__ == "__main__":
    app()