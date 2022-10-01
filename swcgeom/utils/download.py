"""Download helpers."""

import itertools
import logging
import multiprocessing
import os
from urllib.parse import urljoin

import bs4
import urllib3
import urllib3.exceptions

__all__ = ["download_file", "fetch_page", "clone_index_page"]


def download_file(dist: str, url: str) -> None:
    """Download a file."""
    conn = urllib3.connection_from_url(url)
    r = conn.request("GET", url)

    dirname = os.path.dirname(dist)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(dist, "wb") as file:
        file.write(r.data)


def fetch_page(url: str) -> bs4.BeautifulSoup:
    """Get page."""
    conn = urllib3.connection_from_url(url)
    r = conn.request("GET", url)
    data = r.data.decode("utf-8")
    return bs4.BeautifulSoup(data)


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

    files = get_urls_in_index_page(index_url)
    logging.info("downloader: search `{}`, found {} files.", index_url, len(files))

    def download(url: str) -> None:
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
        p.map(download, files)


def get_urls_in_index_page(url: str) -> list[str]:
    """Get all file links by dfs."""
    soup = fetch_page(url)
    links = [el.attrs["href"] for el in soup.find_all("a")]
    files = [urljoin(url, a) for a in links if not a.endswith("/")]
    dirs = [urljoin(url, a) for a in links if a != "../" and a.endswith("/")]
    files.extend(itertools.chain(*[get_urls_in_index_page(dir) for dir in dirs]))
    return files
