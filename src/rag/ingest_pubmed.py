"""
PubMed ingestion via NCBI E-utilities (esearch + efetch).

Queries the configured search terms (config.PUBMED_QUERY_TERMS), fetches
abstracts + metadata, and yields structured dicts ready for chunking.

NCBI rate limits: 3 req/s without key, 10 req/s with key.
We throttle to 3 req/s by default (NCBI_API_KEY is optional).

Yields dicts:
  {pmid, title, abstract, doi, pub_date, journal, authors}
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET

import requests

import config

_ESEARCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_BATCH    = 100      # PMIDs per efetch call
_SLEEP    = 0.35     # seconds between requests (< 3/s without key)


def _common_params() -> dict:
    p = {"db": "pubmed", "retmode": "xml"}
    if config.NCBI_API_KEY:
        p["api_key"] = config.NCBI_API_KEY
    return p


def _search_pmids(term: str, max_results: int) -> list[str]:
    """Return up to max_results PMIDs for the search term."""
    params = {
        **_common_params(),
        "term":    term,
        "retmax":  max_results,
        "usehistory": "n",
    }
    r = requests.get(_ESEARCH, params=params, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    pmids = [id_el.text for id_el in root.findall(".//Id")]
    return pmids


def _fetch_records(pmids: list[str]) -> list[dict]:
    """Fetch and parse abstract records for a batch of PMIDs."""
    params = {
        **_common_params(),
        "id":     ",".join(pmids),
        "rettype": "abstract",
    }
    r = requests.get(_EFETCH, params=params, timeout=60)
    r.raise_for_status()
    return _parse_efetch_xml(r.text)


def _parse_efetch_xml(xml_text: str) -> list[dict]:
    """Extract structured fields from an efetch XML response."""
    records = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return records

    for article in root.findall(".//PubmedArticle"):
        rec = _parse_article(article)
        if rec and rec.get("abstract"):
            records.append(rec)
    return records


def _parse_article(article) -> dict | None:
    try:
        medline = article.find("MedlineCitation")
        if medline is None:
            return None

        pmid_el = medline.find("PMID")
        pmid    = pmid_el.text if pmid_el is not None else ""

        art    = medline.find("Article")
        if art is None:
            return None

        # Title
        title_el = art.find("ArticleTitle")
        title    = (title_el.text or "").strip() if title_el is not None else ""

        # Abstract (concatenate all AbstractText parts)
        abstract_parts = art.findall(".//AbstractText")
        abstract = " ".join(
            (el.text or "").strip()
            for el in abstract_parts
            if el.text
        )

        # Journal
        journal_el = art.find(".//Title")
        journal    = (journal_el.text or "").strip() if journal_el is not None else ""

        # Pub date (year only)
        year_el = art.find(".//PubDate/Year")
        if year_el is None:
            year_el = art.find(".//PubDate/MedlineDate")
        pub_date = (year_el.text or "")[:4] if year_el is not None else ""

        # DOI
        doi = ""
        for id_el in article.findall(".//ArticleId"):
            if id_el.get("IdType") == "doi":
                doi = (id_el.text or "").strip()
                break

        # Authors (last names only for brevity)
        author_els = art.findall(".//Author/LastName")
        authors = ", ".join(el.text for el in author_els if el.text)

        return {
            "pmid":     pmid,
            "title":    title,
            "abstract": abstract,
            "doi":      doi,
            "pub_date": pub_date,
            "journal":  journal,
            "authors":  authors,
        }
    except Exception:
        return None


def fetch_all(
    terms: list[str] | None = None,
    max_per_term: int | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Fetch abstracts for all configured search terms.

    Parameters
    ----------
    terms:        override config.PUBMED_QUERY_TERMS
    max_per_term: override config.PUBMED_MAX_RESULTS_PER_TERM
    verbose:      print progress to stdout

    Returns
    -------
    Deduplicated list of abstract dicts (unique by PMID).
    """
    terms       = terms        or config.PUBMED_QUERY_TERMS
    max_per_term = max_per_term or config.PUBMED_MAX_RESULTS_PER_TERM

    seen_pmids: set[str] = set()
    all_records: list[dict] = []

    for term in terms:
        if verbose:
            print(f"[PubMed] Searching: '{term}' (max {max_per_term}) …")
        try:
            pmids = _search_pmids(term, max_per_term)
        except Exception as e:
            print(f"  esearch failed for '{term}': {e}")
            continue

        new_pmids = [p for p in pmids if p not in seen_pmids]
        if verbose:
            print(f"  Found {len(pmids)} PMIDs ({len(new_pmids)} new)")

        for i in range(0, len(new_pmids), _BATCH):
            batch = new_pmids[i : i + _BATCH]
            time.sleep(_SLEEP)
            try:
                records = _fetch_records(batch)
                for r in records:
                    if r["pmid"] not in seen_pmids:
                        seen_pmids.add(r["pmid"])
                        all_records.append(r)
            except Exception as e:
                print(f"  efetch failed for batch {i}–{i+_BATCH}: {e}")

        if verbose:
            print(f"  Running total: {len(all_records)} abstracts")

    return all_records
