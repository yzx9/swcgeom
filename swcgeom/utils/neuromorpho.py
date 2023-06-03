"""NeuroMorpho.org."""

import argparse
import io
import json
import logging
import math
import os
import urllib.parse
from typing import Any, Dict, Optional

import certifi
import lmdb
import pycurl
from tqdm import tqdm

URL_NEURON = "https://neuromorpho.org/api/neuron"
URL_CNG_VERSION = (
    "https://neuromorpho.org/dableFiles/$ARCHIVE/CNG%20version/$NEURON.CNG.swc"
)
KB = 1024
MB = 1024 * KB
GB = 1024 * MB
TB = 1024 * GB

__all__ = ["download_neuromorpho"]


def download_neuromorpho(path: str, **kwargs) -> None:
    download_metadata(os.path.join(path, "metadata"), **kwargs)
    download_cng_version(os.path.join(path, "cng_version"), **kwargs)


def download_metadata(path: str, verbose: bool = False, **kwargs):
    """Download all neuron metadata.

    Parameters
    ----------
    path : str
        Path to save data.
    verbose : bool, default False
        Show verbose log.
    **kwargs :
        Forwarding to `get`.
    """
    # TODO: how to cache between versions?

    res = get_metadata(page=0, page_size=1, **kwargs)
    total = res["page"]["totalElements"]
    page_size = 50

    env = lmdb.Environment(path, map_size=GB)
    pages = range(math.ceil(total / page_size))
    for page in tqdm(pages) if verbose else pages:
        try:
            res = get_metadata(page, page_size=page_size, **kwargs)
            with env.begin(write=True) as tx:
                for neuron in res["_embedded"]["neuronResources"]:
                    k = str(neuron["neuron_id"]).encode("utf-8")
                    v = json.dumps(neuron).encode("utf-8")
                    tx.put(key=k, value=v)

                tx.commit()
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.warning("fails to get metadata of page %s, err: %s", page, e)

    env.close()


def download_cng_version(
    path: str, metadata: str, *, override: bool = False, verbose: bool = False, **kwargs
):
    """Download GNG version swc.

    Parameters
    ----------
    path : str
        Path to save data.
    metadata : str
        Path to lmdb of metadata.
    override : bool, default False
        Override even exists.
    verbose : bool, default False
        Show verbose log.
    **kwargs :
        Forwarding to `get`.
    """
    env_metadata = lmdb.Environment(metadata)
    env = lmdb.Environment(path, map_size=TB)
    with env_metadata.begin() as tx_metadata:
        entries = tx_metadata.cursor()
        if verbose:
            entries = tqdm(entries, total=env_metadata.stat()["entries"])

        for k, v in entries:
            try:
                with env.begin(write=True) as tx:
                    if not override and tx.get(k) is not None:
                        continue

                    data = json.loads(v.decode("utf-8"))
                    swc = get_cng_version(data, **kwargs)
                    tx.put(key=k, value=swc)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.warning(
                    "fails to get cng version: %s, err: %s", k.decode("utf-8"), e
                )

    env_metadata.close()
    env.close()


def get_metadata(page, page_size: int = 50, **kwargs):
    params = {
        "page": page,
        "size": page_size,
        "sort": "neuron_id,neuron_id,asc",
    }
    query = "&".join([f"{k}={v}" for k, v in params.items()])
    url = f"{URL_NEURON}?{query}"

    s = get(url, **kwargs)
    return json.loads(s)


def get_cng_version(metadata: Dict[str, Any], **kwargs):
    archive = urllib.parse.quote(metadata["archive"].lower())
    neuron = urllib.parse.quote(metadata["neuron_name"])
    url = URL_CNG_VERSION.replace("$ARCHIVE", archive).replace("$NEURON", neuron)
    return get(url, **kwargs)


def get(url: str, *, timeout: int = 2 * 60, proxy: Optional[str] = None):
    buffer = io.BytesIO()
    c = pycurl.Curl()
    c.setopt(pycurl.URL, url)  # initializing the request URL
    c.setopt(pycurl.WRITEDATA, buffer)  # setting options for cURL transfer
    c.setopt(
        pycurl.CAINFO, certifi.where()
    )  # setting the file name holding the certificates
    c.setopt(pycurl.TIMEOUT, timeout)
    if proxy is not None:
        c.setopt(pycurl.PROXY, proxy)
    c.perform()  # perform file transfer
    c.close()  # Ending the session and freeing the resources

    return buffer.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download neuron morphologies from neuromorpho.org"
    )
    parser.add_argument("-o", "--path", type=str)
    parser.add_argument("--proxy", type=str, default=None)
    parser.add_argument("--verbose", type=bool, default=True)
    args = parser.parse_args()

    download_neuromorpho(args.path, proxy=args.proxy, verbose=args.verbose)
