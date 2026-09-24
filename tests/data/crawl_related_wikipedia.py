import wikipediaapi
import pandas as pd
import time
import argparse
from pathlib import Path

def crawl_wikipedia(seed_titles, max_pages=500):
    wiki_wiki = wikipediaapi.Wikipedia('BenoStreamDBTest/1.0', 'en')
    
    visited = {}
    queue = list(seed_titles)
    
    nodes = []
    edges = []
    
    print(f"Crawling Wikipedia starting from: {seed_titles}")
    
    while queue and len(visited) < max_pages:
        title = queue.pop(0)
        if title in visited:
            continue
            
        try:
            page = wiki_wiki.page(title)
            if not page.exists():
                continue
                
            doc_id = page.pageid
            visited[title] = doc_id
            
            # Fetch links
            links = page.links
            linked_titles = list(links.keys())
            
            nodes.append({
                'doc_id': doc_id,
                'title': page.title,
                'url': page.fullurl,
                'text': page.summary
            })
            
            # Add to edges (using title temporarily, will map to id later)
            for linked_title in linked_titles:
                edges.append({'source_title': title, 'target_title': linked_title})
                
                # Add to queue if we need more pages
                if len(visited) + len(queue) < max_pages and linked_title not in visited and linked_title not in queue:
                    queue.append(linked_title)
                    
            print(f"Crawled {len(visited)}/{max_pages}: {title} ({len(linked_titles)} outgoing links)")
            time.sleep(0.05)  # Be nice to Wikipedia API
            
        except Exception as e:
            print(f"Error crawling {title}: {e}")
            
    # Resolve edges
    print(f"Resolving {len(edges)} total edges...")
    valid_edges = []
    for edge in edges:
        if edge['source_title'] in visited and edge['target_title'] in visited:
            valid_edges.append({
                'source': visited[edge['source_title']],
                'target': visited[edge['target_title']],
                'weight': 1.0
            })
            
    nodes_df = pd.DataFrame(nodes)
    edges_df = pd.DataFrame(valid_edges)
    
    print(f"Final Graph: {len(nodes_df)} nodes, {len(edges_df)} edges")
    return nodes_df, edges_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages", type=int, default=500)
    args = parser.parse_args()
    
    seeds = ["Artificial intelligence", "Machine learning", "Data science", "Database", "Graph database"]
    nodes_df, edges_df = crawl_wikipedia(seeds, max_pages=args.max_pages)
    
    output_dir = Path("tests/data")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    nodes_df.to_parquet(output_dir / "organic_wiki_nodes.parquet")
    edges_df.to_parquet(output_dir / "organic_wiki_edges.parquet")
    print("Saved organic_wiki_nodes.parquet and organic_wiki_edges.parquet")
