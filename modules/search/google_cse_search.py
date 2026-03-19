import requests
from utils.cache_utils import load_cache, save_cache


class GoogleCSESearch:

    def __init__(self, api_key, cse_id, max_results, max_urls):
        self.api_key = api_key
        self.cx = cse_id
        self.max_results = max_results
        self.max_urls = max_urls
        self.cache = load_cache()

    def search(self, query):

        cache_key = f"{query}|{self.max_results}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        url = "https://www.googleapis.com/customsearch/v1"

        results = []

        try:
            # 🔥 paginação (10 por vez)
            for start in range(1, self.max_results + 1, 10):

                params = {
                    "key": self.api_key,
                    "cx": self.cx,
                    "q": query,
                    "num": min(10, self.max_results - len(results)),
                    "start": start
                }

                response = requests.get(url, params=params, timeout=10)
                data = response.json()

                items = data.get("items", [])

                for item in items:
                    link = item.get("link")
                    if link:
                        results.append(link)

                # para cedo se já atingiu limite
                if len(results) >= self.max_results:
                    break

        except Exception:
            return []

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