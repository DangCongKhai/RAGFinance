import logging 
from time import time
from functools import wraps


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(filename='process.log')
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def timer(func):
    @wraps(func)
    def measure_time(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        total_time_taken = t2 - t1
        #  Things to log: Loading docoment time + Retrieval time + Reranking time
        if 'retrieve' in func.__name__:
            top_k = kwargs.get('top_k', None)
            queries = kwargs['queries']
            query_len = len(queries)
            if top_k is None:
                top_k = args[-1]
            logger.info(f"Retrieval time for {query_len} queries, {top_k} docs/query= {round(total_time_taken, 4)}s")
        else:
            logger.info(f"Func '{measure_time.__name__}' - Time taken = {round(total_time_taken, 4)}s")
        return result
    return measure_time








