import os
import argparse
import json

moviefile = os.path.join('data', 'movies.json')

def keywordsearch(data, query):
    results = []
    for movie in data:
        if query in movie['title']:
            results.append(movie)
    return results
        


def formatresults(results):
    results.sort(key=lambda x: x['id'])
    for i in range(len(results)):
        print(f"{i+1}. {results[i]['title']}")

def loadFromJson(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f'Searching for: {args.query}')
            data = loadFromJson(moviefile)
            results = keywordsearch(data['movies'], args.query)
            formatresults(results)
            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()