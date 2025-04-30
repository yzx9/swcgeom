# SPDX-FileCopyrightText: 2022 - 2025 Zexin Yuan <pypi@yzx9.xyz>
#
# SPDX-License-Identifier: Apache-2.0

"""Download helpers.

NOTE: All denpendencies need to be installed, try:

```sh
pip install swcgeom[all]
```
"""

import itertools
import logging
import multiprocessing
import os
from functools import partial
from urllib.parse import urljoin

__all__ = ["download", "fetch_page", "clone_index_page"]


def download(dst: str, url: str) -> None:
    """Download a file."""
    from urllib3 import connection_from_url

    conn = connection_from_url(url)
    r = conn.request("GET", url)

    dirname = os.path.dirname(dst)
    if dirname != "" and not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(dst, "wb") as file:
        file.write(r.data)


def fetch_page(url: str):
    """Fetch page content."""
    from bs4 import BeautifulSoup
    from urllib3 import connection_from_url

    conn = connection_from_url(url)
    r = conn.request("GET", url)
    data = r.data.decode("utf-8")
    return BeautifulSoup(data, features="html.parser")


def clone_index_page(
    index_url: str, dist_dir: str, override: bool = False, multiprocess: int = 4
) -> None:
    """Download directory from index page.

    E.g: `https://download.brainimagelibrary.org/biccn/zeng/luo/fMOST/cells/`

    Args:
        index_url: URL of index page.
        dist_dir: Directory of dist.
        override: Override existing file, skip file if `False`.
        multiprocess: How many process are available for download.
    """
    files = get_urls_in_index_page(index_url)
    logging.info("downloader: search `%s`, found %s files.", index_url, len(files))

    task = partial(
        _clone_index_page, index_url=index_url, dist_dir=dist_dir, override=override
    )
    with multiprocessing.Pool(multiprocess) as p:
        p.map(task, files)


def _clone_index_page(url: str, index_url: str, dist_dir: str, override: bool) -> None:
    from urllib3.exceptions import HTTPError

    filepath = url.removeprefix(index_url)
    dist = os.path.join(dist_dir, filepath)
    if os.path.exists(dist):
        if not override:
            logging.info("downloader: file `%s` exits, skipped.", dist)
            return

        logging.info("downloader: file `%s` exits, deleted.", dist)
        os.remove(dist)

    try:
        logging.info("downloader: downloading `%s` to `%s`", url, dist)
        download(dist, url)
        logging.info("downloader: download `%s` to `%s`", url, dist)
    except HTTPError as ex:
        logging.info("downloader: fails to download `%s`, except `%s`", url, ex)


def get_urls_in_index_page(url: str) -> list[str]:
    """Get all file links by dfs."""
    soup = fetch_page(url)
    links = [el.attrs["href"] for el in soup.find_all("a")]
    files = [urljoin(url, a) for a in links if not a.endswith("/")]
    dirs = [urljoin(url, a) for a in links if a != "../" and a.endswith("/")]
    files.extend(itertools.chain(*[get_urls_in_index_page(dir) for dir in dirs]))
    return files


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download files from index page.")
    parser.add_argument("url", type=str, help="URL of index page.")
    parser.add_argument("dist", type=str, help="Directory of dist.")
    parser.add_argument(
        "--override", type=bool, default=False, help="Override existing file."
    )
    parser.add_argument(
        "--multiprocess", type=int, default=4, help="How many process are available."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    clone_index_page(args.url, args.dist, args.override, args.multiprocess)
