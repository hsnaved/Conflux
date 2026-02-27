from typing import List, Dict
from atlassian import Confluence
from bs4 import BeautifulSoup


def html_to_text(html: str) -> str:
    """Convert Confluence HTML to clean plain text."""
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    return soup.get_text(separator="\n", strip=True)


def fetch_confluence_pages(
    base_url: str,
    username: str,
    api_token: str,
    space_key: str,
    limit: int = 100,
) -> List[Dict[str, str]]:
    """Fetch pages from Confluence and return cleaned text."""

    if not all([base_url, username, api_token, space_key]):
        raise RuntimeError("Confluence configuration missing in environment variables")

    confluence = Confluence(
        url=base_url,
        username=username,
        password=api_token,
        cloud=True,
        timeout=30,
    )

    page_contents: List[Dict[str, str]] = []
    start = 0

    while True:
        pages = confluence.get_all_pages_from_space(
            space=space_key,
            start=start,
            limit=limit,
            status="current",
        )

        if not pages:
            break

        for page in pages:
            page_id = page.get("id")
            title = page.get("title", "")

            try:
                content = confluence.get_page_by_id(
                    page_id,
                    expand="body.storage",
                )

                html_text = content["body"]["storage"]["value"]
                plain_text = html_to_text(html_text)

                page_contents.append(
                    {
                        "id": page_id,
                        "title": title,
                        "content": plain_text,
                    }
                )

            except Exception as e:
                print(f"Failed to fetch page {title} ({page_id}): {e}")

        start += limit

    return page_contents