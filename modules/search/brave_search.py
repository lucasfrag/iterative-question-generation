import requests
from utils.cache_utils import load_cache, save_cache


class BraveSearch:

    def __init__(self, api_key, max_results, max_urls):
        self.api_key = api_key
        self.max_results = max_results
        self.max_urls = max_urls
        self.cache = load_cache()

    def search(self, query):

        # 🔑 chave do cache inclui config (importante!)
        cache_key = f"{query}|{self.max_results}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        url = "https://api.search.brave.com/res/v1/web/search"

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }

        params = {
            "q": query,
            "count": self.max_results
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            data = response.json()
        except Exception:
            return []

        results = []

        for r in data.get("web", {}).get("results", []):
            href = r.get("url")
            if href:
                results.append(href)

        # salvar cache
        self.cache[cache_key] = results
        save_cache(self.cache)

        return results

    def run(self, context):

        queries = [context.claim]

        questions = getattr(context, "questions", [])
        queries.extend(questions[:5])

        # bônus (igual DDG)
        queries.append(f"{context.claim} fact check")

        urls = []

        for q in queries:
            urls.extend(self.search(q))

        unique_urls = list(set(urls))

        context.search_results = unique_urls[:self.max_urls]

        return context