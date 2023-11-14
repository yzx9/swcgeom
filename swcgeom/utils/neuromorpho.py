"""NeuroMorpho.org.

Examples
--------

Metadata: 

```json
{
    'neuron_id': 1,
    'neuron_name': 'cnic_001',
    'archive': 'Wearne_Hof',
    'note': 'When originally released, this reconstruction had been incompletely processed, and this issue was fixed in release 6.1 (May 2015). The pre-6.1 version of the processed file is available for download <a href=" dableFiles/previous/v6.1/wearne_hof/cnic_001.CNG.swc ">here</a>.',
    'age_scale': 'Year',
    'gender': 'Male/Female',
    'age_classification': 'old',
    'brain_region': ['neocortex', 'prefrontal', 'layer 3'],
    'cell_type': ['Local projecting', 'pyramidal', 'principal cell'],
    'species': 'monkey',
    'strain': 'Rhesus',
    'scientific_name': 'Macaca mulatta',
    'stain': 'lucifer yellow',
    'experiment_condition': ['Control'],
    'protocol': 'in vivo',
    'slicing_direction': 'custom',
    'reconstruction_software': 'Neurozoom',
    'objective_type': 'Not reported',
    'original_format': 'Neurozoom.swc',
    'domain': 'Dendrites, Soma, No Axon',
    'attributes': 'Diameter, 3D, Angles',
    'magnification': '100',
    'upload_date': '2006-08-01',
    'deposition_date': '2005-12-31',
    'shrinkage_reported': 'Reported',
    'shrinkage_corrected': 'Not Corrected',
    'reported_value': None,
    'reported_xy': None,
    'reported_z': None,
    'corrected_value': None,
    'corrected_xy': None,
    'corrected_z': None,
    'soma_surface': '834.0',
    'surface': '8842.91',
    'volume': '4725.89',
    'slicing_thickness': '400',
    'min_age': '24.0',
    'max_age': '25.0',
    'min_weight': '4500.0',
    'max_weight': '10000.0',
    'png_url': 'http://neuromorpho.org/images/imageFiles/Wearne_Hof/cnic_001.png',
    'reference_pmid': ['12204204', '12902394'],
    'reference_doi': ['10.1016/S0306-4522(02)00305-6', '10.1093/cercor/13.9.950'],
    'physical_Integrity': 'Dendrites Moderate',
    '_links': {
        'self': {
            'href': 'http://neuromorpho.org/api/neuron/id/1'
        },
        'measurements': {
            'href': 'http://neuromorpho.org/api/morphometry/id/1'
        },
        'persistence_vector': {
            'href': 'http://neuromorpho.org/api/pvec/id/1'
        }
    }
}
```

Notes
-----
All denpendencies need to be installed, try:

```sh
pip install swcgeom[all]
```
"""

import argparse
import io
import json
import logging
import math
import os
import urllib.parse
from typing import Any, Callable, Dict, Iterable, List, Optional

from swcgeom.utils import FileReader

__all__ = [
    "neuromorpho_is_valid",
    "neuromorpho_convert_lmdb_to_swc",
    "download_neuromorpho",
]

URL_NEURON = "https://neuromorpho.org/api/neuron"
URL_CNG_VERSION = (
    "https://neuromorpho.org/dableFiles/$ARCHIVE/CNG%20version/$NEURON.CNG.swc"
)
API_NEURON_MAX_SIZE = 500

KB = 1024
MB = 1024 * KB
GB = 1024 * MB

# Test version: 8.5.25 (2023-08-01)
# About 1.1 GB and 18 GB
# No ETAs for future version
SIZE_METADATA = 2 * GB
SIZE_DATA = 20 * GB

# fmt:off
# Test version: 8.5.25 (2023-08-01)
# No ETAs for future version
invalid_ids = [
    # bad file
    81062, 86970, 79791,

    33294, # bad tree with multi root
    268441, # invalid type `-1` in L5467

    # # 404 not found
    # # We don't mark these ids, since they will throw a warning when
    # # downloading and converting, so that users can find out as early
    # # as possible, and can recover immediately when the website fixes
    # # this problem.
    # 97058, 98302, 125801, 130581, 267258, 267259, 267261, 267772,
    # 267773, 268284, 268285, 268286
]
# fmt: on


def neuromorpho_is_valid(metadata: Dict[str, Any]) -> bool:
    return metadata["neuron_id"] not in invalid_ids


# pylint: disable-next=too-many-locals
def neuromorpho_convert_lmdb_to_swc(
    root: str,
    dest: Optional[str] = None,
    *,
    group_by: Optional[str | Callable[[Dict[str, Any]], str | None]] = None,
    where: Optional[Callable[[Dict[str, Any]], bool]] = None,
    encoding: str | None = "utf-8",
    verbose: bool = False,
) -> None:
    """Convert lmdb format to SWCs.

    Parameters
    ----------
    path : str
    dest : str, optional
        If None, use `path/swc`.
    group_by : str | (metadata: Dict[str, Any]) -> str | None, optional
        Group neurons by metadata. If a None is returned then no
        grouping. If a string is entered, use it as a metadata
        attribute name for grouping, e.g.: `archive`, `species`.
    where : (metadata: Dict[str, Any]) -> bool, optional
        Filter neurons by metadata.
    encoding : str | None, default to `utf-8`
        Change swc encoding, part of the original data is not utf-8
        encoded. If is None, keep the original encoding format.
    verbose : bool, default False
        Print verbose info.

    Notes
    -----
    We are asserting the following folder.

    ```text
    |- root
    | |- metadata       # input
    | |- cng_version    # input
    | |- swc            # output
    | | |- groups       # output of groups if grouped
    ```

    See Also
    --------
    neuromorpho_is_valid :
        Recommended filter function, try `where=neuromorpho_is_valid`
    """
    import lmdb
    from tqdm import tqdm

    assert os.path.exists(root)

    env_m = lmdb.Environment(os.path.join(root, "metadata"), readonly=True)
    with env_m.begin() as tx_m:
        where = where or (lambda _: True)
        if isinstance(group_by, str):
            key = group_by
            group_by = lambda v: v[key]  # pylint: disable=unnecessary-lambda-assignment
        elif group_by is None:
            group_by = lambda _: None  # pylint: disable=unnecessary-lambda-assignment
        items = []
        for k, v in tx_m.cursor():
            metadata = json.loads(v)
            if where(metadata):
                items.append((k, group_by(metadata)))

    env_m.close()

    dest = dest or os.path.join(root, "swc")
    os.makedirs(dest, exist_ok=True)
    for grp in set(grp for _, grp in items if grp is not None):
        os.makedirs(os.path.join(dest, grp), exist_ok=True)

    env_c = lmdb.Environment(os.path.join(root, "cng_version"), readonly=True)
    with env_c.begin() as tx_c:
        for k, grp in tqdm(items) if verbose else items:
            kk = k.decode("utf-8")
            try:
                bs = tx_c.get(k)
                if bs is None:
                    logging.warning("cng version of '%s' not exists", kk)
                    continue

                fs = (
                    os.path.join(dest, grp, f"{kk}.swc")
                    if grp is not None
                    else os.path.join(dest, f"{kk}.swc")
                )

                if encoding is None:
                    with open(fs, "wb") as f:
                        f.write(bs)  # type: ignore
                else:
                    bs = io.BytesIO(bs)  # type: ignore
                    with (
                        open(fs, "w", encoding=encoding) as fw,
                        FileReader(bs, encoding="detect") as fr,
                    ):
                        fw.writelines(fr.readlines())
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.warning("fails to convert of %s, err: %s", kk, e)

    env_c.close()


def download_neuromorpho(
    path: str, *, retry: int = 3, verbose: bool = False, **kwargs
) -> None:
    kwargs.setdefault("verbose", verbose)

    path_m = os.path.join(path, "metadata")
    path_c = os.path.join(path, "cng_version")

    err_pages = download_metadata(path_m, **kwargs)
    for i in range(retry):
        if len(err_pages) == 0:
            break

        log = print if verbose else logging.info
        log("retry %d of download metadata: %s", i, json.dumps(err_pages))
        err_pages = download_metadata(path_m, pages=err_pages, **kwargs)

    if len(err_pages) != 0:
        logging.warning(
            "download metadata pages failed after %d retry: %s",
            retry,
            json.dumps(err_pages),
        )

    err_keys = download_cng_version(path_c, path_m, **kwargs)
    for i in range(retry):
        if len(err_keys) == 0:
            break

        err_keys_str = json.dumps([i.decode("utf-8") for i in err_keys])
        logging.info("retry %d download CNG version: %d", i, err_keys_str)
        if verbose:
            print(f"retry {i} download CNG version: {err_keys_str}")
        err_keys = download_cng_version(path_c, path_m, keys=err_keys, **kwargs)

    if len(err_keys) != 0:
        err_keys_str = json.dumps([i.decode("utf-8") for i in err_keys])
        logging.warning(
            "download CNG version failed after %d retry: %s", retry, err_keys_str
        )


def download_metadata(
    path: str, *, pages: Optional[Iterable[int]] = None, verbose: bool = False, **kwargs
) -> List[int]:
    """Download all neuron metadata.

    Parameters
    ----------
    path : str
        Path to save data.
    pages : list of int, optional
        If is None, download all pages.
    verbose : bool, default False
        Show verbose log.
    **kwargs :
        Forwarding to `get`.

    Returns
    -------
    err_pages : list of int
        Failed pages.
    """
    # TODO: how to cache between versions?
    import lmdb
    from tqdm import tqdm

    env = lmdb.Environment(path, map_size=SIZE_METADATA)
    page_size = API_NEURON_MAX_SIZE
    if pages is None:
        res = get_metadata(page=0, page_size=1, **kwargs)
        total = res["page"]["totalElements"]
        pages = range(math.ceil(total / page_size))

    err_pages = []
    for page in tqdm(pages) if verbose else pages:
        try:
            res = get_metadata(page, page_size=page_size, **kwargs)
            with env.begin(write=True) as tx:
                for neuron in res["_embedded"]["neuronResources"]:
                    k = str(neuron["neuron_id"]).encode("utf-8")
                    v = json.dumps(neuron).encode("utf-8")
                    tx.put(key=k, value=v)
        except Exception as e:  # pylint: disable=broad-exception-caught
            err_pages.append(page)
            logging.warning("fails to get metadata of page %s, err: %s", page, e)

    env.close()
    return err_pages


# pylint: disable-next=too-many-locals
def download_cng_version(
    path: str,
    path_metadata: str,
    *,
    keys: Optional[Iterable[bytes]] = None,
    override: bool = False,
    verbose: bool = False,
    **kwargs,
) -> List[bytes]:
    """Download GNG version swc.

    Parameters
    ----------
    path : str
        Path to save data.
    path_metadata : str
        Path to lmdb of metadata.
    keys : list of bytes, optional
        If exist, ignore `override` option. If None, download all key.
    override : bool, default False
        Override even exists.
    verbose : bool, default False
        Show verbose log.
    **kwargs :
        Forwarding to `get`.

    Returns
    -------
    err_keys : list of str
        Failed keys.
    """
    import lmdb
    from tqdm import tqdm

    env_m = lmdb.Environment(path_metadata, map_size=SIZE_METADATA, readonly=True)
    env_c = lmdb.Environment(path, map_size=SIZE_DATA)
    if keys is None:
        with env_m.begin() as tx_m:
            if override:
                keys = [k for k, v in tx_m.cursor()]
            else:
                with env_c.begin() as tx:
                    keys = [k for k, v in tx_m.cursor() if tx.get(k) is None]

    err_keys = []
    for k in tqdm(keys) if verbose else keys:
        try:
            with env_m.begin() as tx:
                metadata = json.loads(tx.get(k).decode("utf-8"))  # type: ignore

            swc = get_cng_version(metadata, **kwargs)
            with env_c.begin(write=True) as tx:
                tx.put(key=k, value=swc)
        except Exception as e:  # pylint: disable=broad-exception-caught
            err_keys.append(k)
            logging.warning(
                "fails to get cng version of '%s', err: %s", k.decode("utf-8"), e
            )

    env_m.close()
    env_c.close()
    return err_keys


def get_metadata(
    page, page_size: int = API_NEURON_MAX_SIZE, **kwargs
) -> Dict[str, Any]:
    params = {
        "page": page,
        "size": page_size,
        "sort": "neuron_id,neuron_id,asc",
    }
    query = "&".join([f"{k}={v}" for k, v in params.items()])
    url = f"{URL_NEURON}?{query}"

    s = get(url, **kwargs)
    return json.loads(s)


def get_cng_version(metadata: Dict[str, Any], **kwargs) -> bytes:
    """Get CNG version swc.

    Returns
    -------
    bs : bytes
        SWC bytes, encoding is NOT FIXED.
    """
    archive = urllib.parse.quote(metadata["archive"].lower())
    neuron = urllib.parse.quote(metadata["neuron_name"])
    url = URL_CNG_VERSION.replace("$ARCHIVE", archive).replace("$NEURON", neuron)
    return get(url, **kwargs)


def get(url: str, *, timeout: int = 2 * 60, proxy: Optional[str] = None) -> bytes:
    # pylint: disable=c-extension-no-member
    import certifi
    import pycurl

    buffer = io.BytesIO()
    c = pycurl.Curl()
    c.setopt(pycurl.URL, url)
    c.setopt(pycurl.WRITEDATA, buffer)
    c.setopt(pycurl.CAINFO, certifi.where())
    c.setopt(pycurl.TIMEOUT, timeout)
    if proxy is not None:
        c.setopt(pycurl.PROXY, proxy)
    c.perform()

    code = c.getinfo(pycurl.RESPONSE_CODE)
    if code != 200:
        raise ConnectionError(f"fails to fetch data, status: {code}")

    c.close()
    return buffer.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use data from neuromorpho.org")
    subparsers = parser.add_subparsers(required=True)

    sub = subparsers.add_parser("download")
    sub.add_argument("-o", "--path", type=str)
    sub.add_argument("--retry", type=int, default=3)
    sub.add_argument("--proxy", type=str, default=None)
    sub.add_argument("--verbose", type=bool, default=True)
    sub.set_defaults(func=download_neuromorpho)

    sub = subparsers.add_parser("convert")
    sub.add_argument("-i", "--root", type=str, required=True)
    sub.add_argument("-o", "--dest", type=str, default=None)
    sub.add_argument("--group_by", type=str, default=None)
    sub.add_argument("--encoding", type=str, default="utf-8")
    sub.add_argument("--verbose", type=bool, default=True)
    sub.set_defaults(func=neuromorpho_convert_lmdb_to_swc)

    args = parser.parse_args()
    func = args.func
    del args.func  # type: ignore
    func(**vars(args))
