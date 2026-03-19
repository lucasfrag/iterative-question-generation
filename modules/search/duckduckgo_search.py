from ddgs import DDGS


class DuckDuckGoSearch:

    def __init__(self, max_results, max_urls):
        # sem default → força uso correto via factory
        self.max_results = max_results
        self.max_urls = max_urls

    def search(self, query: str):

        results = []

        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=self.max_results):
                    href = r.get("href")
                    if href:
                        results.append(href)

        except Exception:
            # ideal: logar erro depois
            return []

        return results

    def run(self, context):

        queries = [context.claim] + getattr(context, "questions", [])

        urls = []

        for q in queries:
            urls.extend(self.search(q))

        # 🔑 deduplicação + corte controlado
        unique_urls = list(set(urls))

        context.search_results = unique_urls[:self.max_urls]

        return context