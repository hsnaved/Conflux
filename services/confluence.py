from atlassian import Confluence
from bs4 import BeautifulSoup


def html_to_text(html: str) -> str:
    """Convert HTML to plain text."""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def fetch_confluence_pages(base_url: str, username: str, api_token: str, space_key: str) -> list[dict]:
    """Fetch all pages from a Confluence space and extract their text content."""
    confluence = Confluence(
        url=base_url,
        username=username,
        password=api_token
    )
    pages = confluence.get_all_pages_from_space(space=space_key, start=0, limit=100, status='current')
    page_contents = []
    for page in pages:
        content = confluence.get_page_by_id(page['id'], expand='body.storage')
        html_text = content['body']['storage']['value']
        plain_text = html_to_text(html_text)
        page_contents.append({'title': page['title'], 'content': plain_text})
    return page_contents