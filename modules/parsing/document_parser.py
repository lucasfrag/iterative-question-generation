import trafilatura

from utils.page_cache import load_page, save_page


class DocumentParser:

    def run(self, context):

        documents = []

        for url in context.search_results:
            cached = load_page(url)

            if cached:
                #print("PAGE CACHE HIT:", url)
                documents.append(cached)
                continue

            try:
                downloaded = trafilatura.fetch_url(url)
                text = trafilatura.extract(downloaded)

                if text:
                    save_page(url, text)
                    documents.append(text)

            except:
                continue

        context.documents = documents

        return context