import time
from utils.helpers import get_citation_data
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

def rank_papers(papers, user_query, use_citations=True, external_progress=None):
    """
    Rank papers based on multiple signals including relevance and citations.
    
    Args:
        papers (list): List of paper dictionaries
        user_query (str): Original user query
        use_citations (bool): Whether to incorporate citation data
        external_progress: An existing Progress object to use instead of creating a new one
        
    Returns:
        list: List of paper dictionaries ranked by combined score
    """
    if not papers:
        return []
    
    # Calculate text similarity scores
    abstracts = [paper['abstract'] for paper in papers]
    titles = [paper['title'] for paper in papers]
    
    # Create TF-IDF vectorizers
    abstract_vectorizer = TfidfVectorizer(stop_words='english')
    title_vectorizer = TfidfVectorizer(stop_words='english')
    
    # Fit and transform
    abstract_matrix = abstract_vectorizer.fit_transform(abstracts + [user_query])
    title_matrix = title_vectorizer.fit_transform(titles + [user_query])
    
    # Calculate similarities
    abstract_similarities = cosine_similarity(abstract_matrix[-1], abstract_matrix[:-1]).flatten()
    title_similarities = cosine_similarity(title_matrix[-1], title_matrix[:-1]).flatten()
    
    # Fetch citation data if requested
    if use_citations:
        citation_task = None
        
        # Use the provided progress object or create a new one
        if external_progress:
            # Complete the previous task before adding a new one for citations
            initial_tasks = external_progress.tasks
            if initial_tasks:
                for task in initial_tasks:
                    external_progress.update(task.id, completed=1.0)
            
            # Create a task in the external progress object
            citation_task = external_progress.add_task("[bold magenta]Fetching citation data...[/bold magenta]", total=len(papers))
            progress_obj = external_progress
        else:
            # Create our own progress object
            progress_obj = Progress(
                SpinnerColumn(),
                TextColumn("[bold magenta]Fetching citation data...[/bold magenta]"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                transient=True
            )
            progress_obj.__enter__()  # Start the progress context
            citation_task = progress_obj.add_task("Fetching citations", total=len(papers))
        
        try:
            # Fetch citations with progress updates
            for i, paper in enumerate(papers):
                try:
                    progress_obj.update(citation_task, completed=i+1, 
                                      description=f"[bold magenta]Fetching citation data...[/bold magenta] Paper {i+1}/{len(papers)}")
                    papers[i]['citation_data'] = get_citation_data(paper['id'])
                except Exception as e:
                    papers[i]['citation_data'] = {'citation_count': 0, 'error': f"[bold orange1]Failed to fetch: {str(e)}[/bold orange1]"}
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
        finally:
            # Only exit the context if we created our own progress object
            if not external_progress:
                progress_obj.__exit__(None, None, None)  # Close the progress context
    
    # Calculate combined scores
    scored_papers = []
    for i, paper in enumerate(papers):
        # Base relevance score (weighted combination of title and abstract similarity)
        relevance_score = (0.6 * title_similarities[i]) + (0.4 * abstract_similarities[i])
        
        # Citation score (normalized)
        citation_score = 0
        if use_citations and 'citation_data' in paper:
            # Get the max citation count for normalization
            max_citations = max([p.get('citation_data', {}).get('citation_count', 0) for p in papers]) or 1
            citation_count = paper['citation_data'].get('citation_count', 0)
            
            # Normalize citation count
            citation_score = citation_count / max_citations
        
        # Recency score (favor newer papers slightly)
        import datetime
        current_year = datetime.datetime.now().year
        pub_year = int(paper['published'][:4])
        recency_score = 1 - ((current_year - pub_year) / 10)  # Decay over 10 years
        recency_score = max(0, min(1, recency_score))  # Clamp between 0 and 1
        
        # Combined score with weights
        combined_score = (0.5 * relevance_score) + (0.4 * citation_score) + (0.1 * recency_score)
        
        # Store the scores for explanation purposes
        paper['scores'] = {
            'relevance': relevance_score,
            'citation': citation_score,
            'recency': recency_score,
            'combined': combined_score
        }
        
        scored_papers.append((paper, combined_score))
    
    # Sort by combined score (descending)
    scored_papers.sort(key=lambda x: x[1], reverse=True)
    
    # Return the ranked papers
    return [paper for paper, _ in scored_papers]