from config import Config

from modules.search.duckduckgo_search import DuckDuckGoSearch


def get_searcher():

    engine = Config.SEARCH_ENGINE

    if engine == "duckduckgo":
        return DuckDuckGoSearch(
            max_results=Config.SEARCH_MAX_RESULTS,
            max_urls=Config.SEARCH_MAX_URLS,
            max_workers=Config.SEARCH_MAX_WORKERS
        )

    # fallback
    return DuckDuckGoSearch(
        max_results=Config.SEARCH_MAX_RESULTS,
        max_urls=Config.SEARCH_MAX_URLS,
    )