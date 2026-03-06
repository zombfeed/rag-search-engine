import os
import string
import argparse
import json

from nltk.stem import PorterStemmer

moviefile = os.path.join('data', 'movies.json')
stopwordsfile = os.path.join('data', 'stopwords.txt')



def keywordsearch(data, query):
    results = []
    punctable = str.maketrans('','',string.punctuation,)
    with open(stopwordsfile, 'r') as f:
        stopwords = f.read().splitlines()
    stemmer = PorterStemmer()
    for movie in data:
        titletoken = list(filter(lambda t: t!='' or t not in stopwords, movie['title'].lower().translate(punctable).split()))
        qtokens = list(filter(lambda t: t != '' or t not in stopwords, query.lower().translate(punctable).split()))
        for token in qtokens:
            tokenroot = stemmer.stem(token)
            for title in titletoken:
                titleroot = stemmer.stem(title)
                if tokenroot in titleroot:
                    results.append(movie)
                    break
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