# /// script
# description = "Prompt with Markdown files"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[openai]>=0.7.8", "numpy", "python-dotenv"]
# ///
from dotenv import load_dotenv

import daft


@daft.func()
def discover_github_urls(repo_url: str) -> list[str]:
    """
    Discover HTTP URLs for files in a GitHub repository.

    Args:
        repo_url: GitHub repository URL (e.g., "https://github.com/user/repo/tree/branch/path")

    Returns:
        List of raw GitHub URLs for files in the repository
    """
    import json
    import re
    import urllib.request

    # Parse the GitHub URL to extract owner, repo, branch, and path
    # Example: https://github.com/LeCoupa/awesome-cheatsheets/tree/88e5be6e4b01edf6c36c8f78b246c8fba70aa058/languages
    pattern = r"github\.com/([^/]+)/([^/]+)/tree/([^/]+)/(.*)"
    match = re.search(pattern, repo_url)

    if not match:
        return []

    owner, repo, branch, path = match.groups()

    # Use GitHub API to list files
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"

    try:
        req = urllib.request.Request(api_url)
        req.add_header("User-Agent", "Mozilla/5.0")

        with urllib.request.urlopen(req) as response:
            contents = json.loads(response.read().decode())

        # Extract raw URLs for files
        urls = []
        for item in contents:
            if item["type"] == "file":
                # Convert to raw GitHub URL
                raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}/{item['name']}"
                urls.append(raw_url)

        return urls
    except Exception as e:
        print(f"Error fetching GitHub contents: {e}")
        return []


if __name__ == "__main__":
    load_dotenv()
#
# # Discover Markdown Files in your Documents Folder
# df = daft.from_glob_path("https://github.com/LeCoupa/awesome-cheatsheets/tree/88e5be6e4b01edf6c36c8f78b246c8fba70aa058/languages/*.md")
#
#
# df = (
#     df
#     # Create a daft.File column from the path
#     .with_column("file", file(col("path")))
#     # Prompt GPT-5-nano with markdown files as context
#     .with_column(
#         "response",
#         prompt(
#             [lit("What are in the contents of this file? \n"), col("file")],
#             model="gpt-5-nano",
#             provider="openai",
#         )
#     )
# )
# df.show(format="fancy", max_width=80)

print(discover_github_urls("Eventual-Inc/Daft"))
