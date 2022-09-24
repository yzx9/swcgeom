"""Download helpers."""

import itertools
import logging
import multiprocessing
import os
from urllib.parse import urljoin

import bs4
import urllib3
import urllib3.exceptions

__all__ = ["download_file", "clone_index_page"]


def download_file(dist: str, url: str) -> None:
    """Download a file."""

    conn = urllib3.connection_from_url(url)
    r = conn.request("GET", url)

    dirname = os.path.dirname(dist)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(dist, "wb") as file:
        file.write(r.data)


def get_page_soup(url: str) -> bs4.BeautifulSoup:
    """Get page."""
    conn = urllib3.connection_from_url(url)
    r = conn.request("GET", url)
    data = r.data.decode("utf-8")
    return bs4.BeautifulSoup(data)


def get_page_links(soup: bs4.BeautifulSoup) -> list[str]:
    """Get links in page."""
    return [el.attrs["href"] for el in soup.find_all("a")]


def get_all_file_urls(url: str) -> list[str]:
    """Get all file links by dfs."""
    soup = get_page_soup(url)
    links = get_page_links(soup)
    files = [urljoin(url, a) for a in links if not a.endswith("/")]
    dirs = [urljoin(url, a) for a in links if a != "../" and a.endswith("/")]
    files.extend(itertools.chain(*[get_all_file_urls(dir) for dir in dirs]))
    return files


def clone_index_page(
    index_url: str, dist_dir: str, override: bool = False, multiprocess: int = 4
) -> None:
    """Download directory from index page.

    E.g: `https://download.brainimagelibrary.org/biccn/zeng/luo/fMOST/cells/`

    Parameters
    ----------
    index_url : str
        URL of index page.
    dist_dir : str
        Directory of dist.
    override : bool, default `False`
        Override existing file, skip file if `False`.
    multiprocess : int, default `4`
        How many process are available for download.
    """

    all_file_urls = get_all_file_urls(index_url)
    logging.info(
        "downloader: search `{}`, found {} files.", index_url, len(all_file_urls)
    )

    def downloader(url: str) -> None:
        filepath = url.removeprefix(index_url)
        dist = os.path.join(dist_dir, filepath)
        if os.path.exists(filepath):
            if not override:
                logging.info("downloader: file `{}` exits, skiped.", dist)
                return

            logging.info("downloader: file `{}` exits, deleted.", dist)
            os.remove(filepath)

        try:
            logging.info("downloader: downloading `{}` to `{}`", url, dist)
            download_file(filepath, url)
            logging.info("downloader: download `{}` to `{}`", url, dist)
        except urllib3.exceptions.HTTPError as ex:
            logging.info("downloader: fails to download `{}`, except `{}`", url, ex)

    with multiprocessing.Pool(multiprocess) as p:
        p.map(downloader, all_file_urls)
