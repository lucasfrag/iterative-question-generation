import requests

from utils.cache_utils import load_cache, save_cache


class WebSearch:

    def __init__(self, api_key):
        self.api_key = api_key
        self.cache = load_cache()

    def search(self, query):

        # 1️⃣ verificar cache

        if query in self.cache:
            #print("CACHE HIT:", query)
            return self.cache[query]

        #print("API SEARCH:", query)

        url = "https://api.search.brave.com/res/v1/web/search"

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }

        params = {
            "q": query,
            "count": 5
        }

        response = requests.get(url, headers=headers, params=params)

        data = response.json()

        results = []

        for r in data.get("web", {}).get("results", []):
            results.append(r["url"])

        # 2️⃣ salvar no cache

        self.cache[query] = results
        save_cache(self.cache)
        return results

    def run(self, context):
        queries = [context.claim] + context.questions
        urls = []

        for q in queries:
            urls.extend(self.search(q))

        context.search_results = list(set(urls))[:10]

        return context