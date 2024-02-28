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
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple

from tqdm import tqdm

from swcgeom.utils import FileReader

__all__ = [
    "neuromorpho_is_valid",
    "neuromorpho_convert_lmdb_to_swc",
    "download_neuromorpho",
    "NeuroMorpho",
]


URL_BASE = "https://neuromorpho.org"
URL_METADATA = "api/neuron"
URL_MORPHO_CNG = "dableFiles/$ARCHIVE/CNG%20version/$NEURON.CNG.swc"
URL_MORPHO_SOURCE = "dableFiles/$ARCHIVE/Source-Version/$NEURON.$EXT"
URL_LOG_CNG = "dableFiles/$ARCHIVE/Remaining%20issues/$NEURON.CNG.swc.std"
URL_LOG_SOURCE = "dableFiles/$ARCHIVE/Standardization%20log/$NEURON.std"
API_PAGE_SIZE_MAX = 500

KB = 1024
MB = 1024 * KB
GB = 1024 * MB

# Test version: 8.5.25 (2023-08-01)
# No ETAs for future version
# Size of metadata about 0.5 GB
# Size of morpho_cng about 18 GB
# Not sure about the size of others
SIZE_METADATA = 2 * GB
SIZE_DATA = 20 * GB

RESOURCES = Literal["morpho_cng", "morpho_source", "log_cng", "log_source"]
DOWNLOAD_CONFIGS: Dict[RESOURCES, Tuple[str, int]] = {
    # name/path: (url, size)
    "morpho_cng": (URL_MORPHO_CNG, 20 * GB),
    "morpho_source": (URL_LOG_CNG, 512 * GB),
    "log_cng": (URL_LOG_CNG, 512 * GB),
    "log_source": (URL_LOG_SOURCE, 512 * GB),
}

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


def neuromorpho_convert_lmdb_to_swc(
    root: str, dest: Optional[str] = None, *, verbose: bool = False, **kwargs
) -> None:
    nmo = NeuroMorpho(root, verbose=verbose)
    nmo.convert_lmdb_to_swc(dest, **kwargs)


def download_neuromorpho(path: str, *, verbose: bool = False, **kwargs) -> None:
    nmo = NeuroMorpho(path, verbose=verbose)
    nmo.download(**kwargs)


class NeuroMorpho:
    def __init__(
        self, root: str, *, url_base: str = URL_BASE, verbose: bool = False
    ) -> None:
        """
        Parameters
        ----------
        root : str
        verbose : bool, default False
            Show verbose log.
        """

        super().__init__()
        self.root = root
        self.url_base = url_base
        self.verbose = verbose

    def download(
        self,
        *,
        retry: int = 3,
        metadata: bool = True,
        resources: Iterable[RESOURCES] = ["morpho_cng"],
        **kwargs,
    ) -> None:
        """Download data from neuromorpho.org."""

        # metadata
        path_m = os.path.join(self.root, "metadata")
        if metadata:
            err_pages = None
            for i in range(retry + 1):
                if err_pages is not None and len(err_pages) == 0:
                    break

                self._info("download metadata")
                if i != 0:
                    self._info("retry %d: %s", i, json.dumps(err_pages))

                err_pages = self._download_metadata(path_m, pages=err_pages, **kwargs)

            self._info("download metadata done")
            if err_pages is not None and len(err_pages) != 0:
                self._warning("fails to download metadata: %s", json.dumps(err_pages))
        else:
            self._info("skip download metadata")

        # file
        def dumps(keys: List[bytes]) -> str:
            return json.dumps([i.decode("utf-8") for i in keys])

        for name in resources:
            url, map_size = DOWNLOAD_CONFIGS[name]
            path = os.path.join(self.root, name)

            err_keys = None
            for i in range(retry + 1):
                if err_keys is not None and len(err_keys) == 0:
                    break

                self._info("download %s", name)
                if err_keys is not None:
                    self._info("retry %d: %s", i, dumps(err_keys))

                err_keys = self._download_files(
                    url, path, path_m, map_size=map_size, **kwargs
                )

            self._info("download %s done", name)
            if err_keys is not None and len(err_keys) != 0:
                self._warning("fails to download %s: %s", name, dumps(err_keys))

    # pylint: disable-next=too-many-locals
    def convert_lmdb_to_swc(
        self,
        dest: Optional[str] = None,
        *,
        group_by: Optional[str | Callable[[Dict[str, Any]], str | None]] = None,
        where: Optional[Callable[[Dict[str, Any]], bool]] = None,
        encoding: str | None = "utf-8",
    ) -> None:
        r"""Convert lmdb format to SWCs.

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
        | |- metadata   # input
        | |- morpho_cng # input
        | |- swc        # output
        | | |- groups   # output of groups if grouped
        ```

        See Also
        --------
        neuromorpho_is_valid :
            Recommended filter function, try `where=neuromorpho_is_valid`
        """

        import lmdb

        env_m = lmdb.Environment(os.path.join(self.root, "metadata"), readonly=True)
        with env_m.begin() as tx_m:
            where = where or (lambda _: True)
            if isinstance(group_by, str):
                key = group_by
                group_by = lambda v: v[
                    key
                ]  # pylint: disable=unnecessary-lambda-assignment
            elif group_by is None:
                group_by = (
                    lambda _: None
                )  # pylint: disable=unnecessary-lambda-assignment
            items = []
            for k, v in tx_m.cursor():
                metadata = json.loads(v)
                if where(metadata):
                    items.append((k, group_by(metadata)))

        env_m.close()

        dest = dest or os.path.join(self.root, "swc")
        os.makedirs(dest, exist_ok=True)
        for grp in set(grp for _, grp in items if grp is not None):
            os.makedirs(os.path.join(dest, grp), exist_ok=True)

        env_c = lmdb.Environment(os.path.join(self.root, "morpho_cng"), readonly=True)
        with env_c.begin() as tx_c:
            for k, grp in tqdm(items) if self.verbose else items:
                kk = k.decode("utf-8")
                try:
                    bs = tx_c.get(k)
                    if bs is None:
                        self._warning("morpho_cng of '%s' not exists", kk)
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
                except (IOError, lmdb.Error) as e:
                    self._warning("fails to convert of %s, err: %s", kk, e)

        env_c.close()

    # Downloader

    def _download_metadata(
        self,
        path: str,
        *,
        pages: Optional[Iterable[int]] = None,
        page_size: int = API_PAGE_SIZE_MAX,
        **kwargs,
    ) -> List[int]:
        r"""Download all neuron metadata.

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

        env = lmdb.Environment(path, map_size=SIZE_METADATA)
        if pages is None:
            res = self._get_metadata(page=0, page_size=1, **kwargs)
            total = res["page"]["totalElements"]
            pages = range(math.ceil(total / page_size))

        err_pages = []
        for page in tqdm(pages) if self.verbose else pages:
            try:
                res = self._get_metadata(page, page_size=page_size, **kwargs)
                with env.begin(write=True) as tx:
                    for neuron in res["_embedded"]["neuronResources"]:
                        k = str(neuron["neuron_id"]).encode("utf-8")
                        v = json.dumps(neuron).encode("utf-8")
                        tx.put(key=k, value=v)
            except IOError as e:
                err_pages.append(page)
                self._warning("fails to get metadata of page %s, err: %s", page, e)

        env.close()
        return err_pages

    # pylint: disable-next=too-many-locals
    def _download_files(
        self,
        url: str,
        path: str,
        path_metadata: str,
        *,
        keys: Optional[Iterable[bytes]] = None,
        override: bool = False,
        map_size: int = 512 * GB,
        **kwargs,
    ) -> List[bytes]:
        """Download files.

        Parameters
        ----------
        url : str
        path : str
            Path to save data.
        path_metadata : str
            Path to lmdb of metadata.
        keys : list of bytes, optional
            If exist, ignore `override` option. If None, download all key.
        override : bool, default False
            Override even exists.
        map_size : int, default 512GB
        **kwargs :
            Forwarding to `get`.

        Returns
        -------
        err_keys : list of str
            Failed keys.
        """

        import lmdb

        env_m = lmdb.Environment(path_metadata, map_size=SIZE_METADATA, readonly=True)
        env_c = lmdb.Environment(path, map_size=map_size)
        if keys is None:
            with env_m.begin() as tx_m:
                if override:
                    keys = [k for k, v in tx_m.cursor()]
                else:
                    with env_c.begin() as tx:
                        keys = [k for k, v in tx_m.cursor() if tx.get(k) is None]

        err_keys = []
        for k in tqdm(keys) if self.verbose else keys:
            try:
                with env_m.begin() as tx:
                    metadata = json.loads(tx.get(k).decode("utf-8"))  # type: ignore

                swc = self._get_file(url, metadata, **kwargs)
                with env_c.begin(write=True) as tx:
                    tx.put(key=k, value=swc)
            except IOError as e:
                err_keys.append(k)
                self._warning(
                    "fails to get morphology file `%s`, err: %s", k.decode("utf-8"), e
                )

        env_m.close()
        env_c.close()
        return err_keys

    def _get_metadata(
        self, page: int, page_size: int = API_PAGE_SIZE_MAX, **kwargs
    ) -> Dict[str, Any]:
        params = {
            "page": page,
            "size": page_size,
            "sort": "neuron_id,neuron_id,asc",
        }
        query = "&".join([f"{k}={v}" for k, v in params.items()])
        url = f"{URL_METADATA}?{query}"
        resp = self._get(url, **kwargs)
        return json.loads(resp)

    def _get_file(self, url: str, metadata: Dict[str, Any], **kwargs) -> bytes:
        """Get file.

        Returns
        -------
        bs : bytes
            Bytes of morphology file, encoding is NOT FIXED.
        """

        archive = urllib.parse.quote(metadata["archive"].lower())
        neuron = urllib.parse.quote(metadata["neuron_name"])
        ext = self._guess_ext(metadata)
        url = (
            url.replace("$ARCHIVE", archive)
            .replace("$NEURON", neuron)
            .replace("$EXT", ext)
        )
        return self._get(url, **kwargs)

    def _get(
        self, url: str, *, timeout: int = 2 * 60, proxy: Optional[str] = None
    ) -> bytes:
        if not url.startswith("http://") and not url.startswith("https://"):
            url = urllib.parse.urljoin(self.url_base, url)

        proxies = None
        if proxy is not None:
            proxies = {"http": proxy, "https": proxy}

        response = self._session().get(url, timeout=timeout, proxies=proxies)
        response.raise_for_status()
        return response.content

    def _session(self) -> Any:
        if hasattr(self, "session"):
            return self.session

        import requests
        import requests.adapters
        import urllib3
        import urllib3.util

        class CustomSSLContextHTTPAdapter(requests.adapters.HTTPAdapter):
            def __init__(self, ssl_context=None, **kwargs):
                self.ssl_context = ssl_context
                super().__init__(**kwargs)

            def init_poolmanager(self, connections, maxsize, block=False):
                super().init_poolmanager(
                    connections, maxsize, block, ssl_context=self.ssl_context
                )

            def proxy_manager_for(self, proxy, **proxy_kwargs):
                return super().proxy_manager_for(
                    proxy, **proxy_kwargs, ssl_context=self.ssl_context
                )

        ctx = urllib3.util.create_urllib3_context()
        ctx.load_default_certs()
        ctx.set_ciphers("DEFAULT@SECLEVEL=1")

        session = requests.session()
        session.adapters.pop("https://", None)
        session.mount("https://", CustomSSLContextHTTPAdapter(ssl_context=ctx))

        self.session = session
        return session

    # format
    def _guess_ext(self, metadata) -> str:
        match metadata["original_format"]:
            case "Custom.xml":
                return "morph.xml"

            case _:
                _, ext = os.path.splitext(metadata["original_format"])
                return ext[1:]

    # log helper

    def _info(self, msg: str, *arg):
        logging.info(msg, *arg, stacklevel=2)
        if self.verbose:
            print(msg.format(*arg))

    def _warning(self, msg: str, *arg):
        logging.warning(msg, *arg, stacklevel=2)
        if self.verbose:
            print(msg.format(*arg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use data from neuromorpho.org")
    subparsers = parser.add_subparsers(required=True)

    sub = subparsers.add_parser("download")
    sub.add_argument("-o", "--path", type=str)
    sub.add_argument("--retry", type=int, default=3)
    sub.add_argument("--metadata", type=bool, default=True)
    sub.add_argument(
        "--resources",
        type=str,
        nargs="*",
        default=["morpho_cng"],
        choices=["morpho_cng", "morpho_source", "log_cng", "log_source"],
    )
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
