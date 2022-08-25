import itertools
import logging
import multiprocessing
import os
from urllib.parse import urljoin

import bs4
import urllib3

_MODULE = "downloader"


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
    """Get all file links by DFS."""
    soup = get_page_soup(url)
    links = get_page_links(soup)
    files = [urljoin(url, a) for a in links if not a.endswith("/")]
    dirs = [urljoin(url, a) for a in links if a != "../" and a.endswith("/")]
    files.extend(itertools.chain(*[get_all_file_urls(dir) for dir in dirs]))
    return files


def download_file(dist_dir: str, url: str) -> None:
    conn = urllib3.connection_from_url(url)
    r = conn.request("GET", url)
    name = url.split("/")[-1]
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)

    with open(os.path.join(dist_dir, name), "wb") as file:
        file.write(r.data)


def download_all(
    index_url: str, dist: str, override: bool = False, multiprocess: int = 4
) -> None:
    """Download directory from index page.

    E.g: `https://download.brainimagelibrary.org/biccn/zeng/luo/fMOST/cells/`

    Parameters
    ==========
    index_url : str.
        URL of index page.
    dist : str.
        Path of target directory.
    override : bool, default to `False`.
        Override existing file, skip file if `False`.
    multiprocess : int, default to 4.
        How many process are available for download.
    """

    all_file_urls = get_all_file_urls(index_url)
    logging.info(f"{_MODULE}: search `{index_url}`, found {len(all_file_urls)} files.")

    def download(dist: str, url: str) -> None:
        file_dist = url.removeprefix(index_url)
        components = file_dist.split("/")[:-1]
        dir = os.path.join(dist, *components)
        if os.path.exists(file_dist):
            if not override:
                return

            logging.info(f"{_MODULE}: file `{dir}` exits, deleted.")
            os.remove(file_dist)

        try:
            logging.info(f"{_MODULE}: downloading `{url}` to `{dir}`")
            download_file(dir, url)

            logging.info(f"{_MODULE}: download `{url}` to `{dir}`")
        except Exception as ex:
            logging.info(f"{_MODULE}: fails to download `{url}`, except `{ex}`")

    with multiprocessing.Pool(multiprocess) as p:
        p.map(lambda url: download(dist, url), all_file_urls)
