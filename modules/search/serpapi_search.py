from serpapi import GoogleSearch
from utils.cache_utils import load_cache, save_cache


class SerpAPISearch:

    def __init__(self, api_key, max_results, max_urls):
        self.api_key = api_key
        self.max_results = max_results
        self.max_urls = max_urls
        self.cache = load_cache()

    def search(self, query):

        # 🔑 cache com config
        cache_key = f"{query}|{self.max_results}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        params = {
            "q": query,
            "api_key": self.api_key,
            "num": self.max_results,   # 🔥 CRÍTICO
        }

        try:
            search = GoogleSearch(params)
            data = search.get_dict()
        except Exception:
            return []

        results = []

        for r in data.get("organic_results", []):
            link = r.get("link")
            if link:
                results.append(link)

        # salvar cache
        self.cache[cache_key] = results
        save_cache(self.cache)

        return results

    def run(self, context):

        queries = [context.claim]

        questions = getattr(context, "questions", [])
        queries.extend(questions[:5])

        # bônus
        queries.append(f"{context.claim} fact check")

        urls = []

        for q in queries:
            urls.extend(self.search(q))

        unique_urls = list(set(urls))

        context.search_results = unique_urls[:self.max_urls]

        return context