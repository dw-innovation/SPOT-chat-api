import os

os.environ.setdefault("SPOT_TAG_SEARCH_URL", "")  # tests mock the tag search API explicitly
os.environ.setdefault("SPOT_DEBUG", "1")         # the API tests look at tool_calls and /health
os.environ.setdefault("SPOT_RATE_LIMIT", "0")
