"""
Lexical feature engineering for Zaxonite Domain Risk ML.

Version 0.1 derives features only from the hostname string.
No DNS, RDAP, website-content or network-derived intelligence
is used in this baseline.
"""

import math
import tldextract

from collections import Counter

TLD_EXTRACTOR = tldextract.TLDExtract(
    suffix_list_urls=()
)


def hostname_length(hostname: str) -> int:
    """Return the total number of characters in a hostname."""
    return len(hostname)


def digit_count(hostname: str) -> int:
    """Return the number of numeric characters in a hostname."""
    return sum(character.isdigit() for character in hostname)


def digit_ratio(hostname: str) -> float:
    """Return the proportion of hostname characters that are numeric."""
    if not hostname:
        return 0.0

    return digit_count(hostname) / hostname_length(hostname)


def hyphen_count(hostname: str) -> int:
    """Return the number of hyphens in a hostname."""
    return hostname.count("-")


def subdomain_depth(hostname: str) -> int:
    """Return the number of subdomain labels before the registrable domain."""
    if not hostname:
        return 0

    extracted = TLD_EXTRACTOR(hostname.strip("."))

    if not extracted.domain or not extracted.suffix:
        return 0

    if not extracted.subdomain:
        return 0

    return len(extracted.subdomain.split("."))


def hostname_entropy(hostname: str) -> float:
    """Return the Shannon entropy of characters in a hostname."""
    if not hostname:
        return 0.0

    counts = Counter(hostname)
    length = len(hostname)

    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )


def alphabetic_ratio(hostname: str) -> float:
    """Return the proportion of hostname characters that are alphabetic."""
    if not hostname:
        return 0.0

    alphabetic_count = sum(
        character.isalpha()
        for character in hostname
    )

    return alphabetic_count / hostname_length(hostname)