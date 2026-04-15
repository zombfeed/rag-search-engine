#!/usr/bin/env python3

import argparse

from lib.semantic_search import(
    verify_model,
    verify_embeddings,
    embed_text,
    embed_query_text,
    semantic_search,
    chunk_text
)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    verify_parser = subparsers.add_parser("verify", help="Display model and max sequence length")
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Display number of docs and shape of embeddings")
    
    embedtext_parser = subparsers.add_parser("embed_text", help="Convert a given text into an embedding")
    embedtext_parser.add_argument('text', type=str, help="Text to embed")
    
    embedquery_parser = subparsers.add_parser("embedquery", help="Convert a given query into an embedding")
    embedquery_parser.add_argument('query', type=str, help="Query to embed")
    
    search_parser = subparsers.add_parser("search", help="Search for movies using semantic search")
    search_parser.add_argument('query', type=str, help="Search query")
    search_parser.add_argument('--limit', type=int, default=5, help="Number of results to return")
    
    chunk_parser = subparsers.add_parser("chunk", help="Chunk a given text into smaller pieces")
    chunk_parser.add_argument('text', type=str, help='Text to chunk')
    chunk_parser.add_argument('--chunk-size', type=int, default=200, help='Maximum number of characters in each chunk')
    chunk_parser.add_argument('--overlap', type=int, help='Number of overlapping characters between chunks', default=0)
    
    
    
    args = parser.parse_args()
    

    match args.command:
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            semantic_search(args.query, args.limit)
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()