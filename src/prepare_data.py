"""
Dataset preparation utilities for Zaxonite Domain Risk ML.

Raw source datasets are read from data/raw/.
Processed datasets are written to data/processed/.

Raw source files must never be modified in place.
"""

from pathlib import Path
from urllib.parse import urlparse
import csv
import ipaddress
import zipfile
import tldextract
import random


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TLD_EXTRACTOR = tldextract.TLDExtract(
    suffix_list_urls=()
)

RANDOM_SEED = 42


def is_ip_address(hostname: str) -> bool:
    """Return True when hostname is an IPv4 or IPv6 address."""

    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def get_registrable_domain(hostname: str) -> str:
    """Return the registrable domain for a hostname."""

    extracted = TLD_EXTRACTOR(hostname)

    if not extracted.domain or not extracted.suffix:
        return ""

    return f"{extracted.domain}.{extracted.suffix}"


def analyse_urlhaus_zip(path: Path) -> None:
    """Analyse hostname composition of a zipped URLhaus CSV dump."""

    total = 0
    ip_hosts = 0
    domain_hosts = 0
    invalid_urls = 0
    unique_domains = set()

    with zipfile.ZipFile(path, "r") as archive:
        with archive.open("csv.txt") as raw_file:
            lines = (
                line.decode("utf-8", errors="replace")
                for line in raw_file
            )

            reader = csv.reader(
                line for line in lines
                if line.strip() and not line.startswith("#")
            )

            for row in reader:
                total += 1

                if len(row) < 3:
                    invalid_urls += 1
                    continue

                url = row[2]

                try:
                    hostname = urlparse(url).hostname
                except ValueError:
                    invalid_urls += 1
                    continue

                if not hostname:
                    invalid_urls += 1
                    continue

                hostname = hostname.lower().rstrip(".")

                if is_ip_address(hostname):
                    ip_hosts += 1
                else:
                    domain_hosts += 1
                    unique_domains.add(hostname)

    print(f"Total observations: {total:,}")
    print(f"IP-host observations: {ip_hosts:,}")
    print(f"Domain-host observations: {domain_hosts:,}")
    print(f"Invalid/unusable URLs: {invalid_urls:,}")
    print(f"Unique domain hostnames: {len(unique_domains):,}")


def analyse_tranco_overlap(urlhaus_path: Path, tranco_path: Path) -> None:
    """Measure overlap between URLhaus domain hosts and Tranco domains."""

    urlhaus_domains = set()

    with zipfile.ZipFile(urlhaus_path, "r") as archive:
        with archive.open("csv.txt") as raw_file:
            lines = (
                line.decode("utf-8", errors="replace")
                for line in raw_file
            )

            reader = csv.reader(
                line for line in lines
                if line.strip() and not line.startswith("#")
            )

            for row in reader:
                if len(row) < 3:
                    continue

                try:
                    hostname = urlparse(row[2]).hostname
                except ValueError:
                    continue

                if not hostname:
                    continue

                hostname = hostname.lower().rstrip(".")

                if not is_ip_address(hostname):
                    urlhaus_domains.add(hostname)

    tranco_domains = set()

    with zipfile.ZipFile(tranco_path, "r") as archive:
        with archive.open("top-1m.csv") as raw_file:
            reader = csv.reader(
                line.decode("utf-8", errors="replace")
                for line in raw_file
            )

            for row in reader:
                if len(row) < 2:
                    continue

                domain = row[1].strip().lower().rstrip(".")

                if domain:
                    tranco_domains.add(domain)

    overlap = urlhaus_domains & tranco_domains

    urlhaus_registrable = {
        get_registrable_domain(domain)
        for domain in urlhaus_domains
    }

    urlhaus_registrable.discard("")

    tranco_registrable = {
        get_registrable_domain(domain)
        for domain in tranco_domains
    }

    tranco_registrable.discard("")

    registrable_overlap = urlhaus_registrable & tranco_registrable

    filtered_urlhaus_domains = {
        domain
        for domain in urlhaus_domains
        if get_registrable_domain(domain) not in registrable_overlap
    }

    print()
    print("Cross-dataset overlap analysis")
    print(f"Unique URLhaus domain hosts: {len(urlhaus_domains):,}")
    print(f"Unique Tranco domains: {len(tranco_domains):,}")
    print(f"Exact hostname overlaps: {len(overlap):,}")
    print(f"Unique URLhaus registrable domains: {len(urlhaus_registrable):,}")
    print(f"Unique Tranco registrable domains: {len(tranco_registrable):,}")
    print(f"Registrable-domain overlaps: {len(registrable_overlap):,}")

    print(
        "URLhaus hostnames remaining after overlap exclusion: "
        f"{len(filtered_urlhaus_domains):,}"
    )

    if overlap:
        print()
        print("Example exact hostname overlaps:")
        for domain in sorted(overlap)[:20]:
            print(f"  {domain}")

    if registrable_overlap:
        print()
        print("Example registrable-domain overlaps:")
        for domain in sorted(registrable_overlap)[:20]:
            print(f"  {domain}")


def build_baseline_dataset(
    urlhaus_path: Path,
    tranco_path: Path,
    output_path: Path,
) -> None:
    """Build a balanced hostname dataset for the v0.1 baseline model."""

    urlhaus_domains = set()

    with zipfile.ZipFile(urlhaus_path, "r") as archive:
        with archive.open("csv.txt") as raw_file:
            lines = (
                line.decode("utf-8", errors="replace")
                for line in raw_file
            )

            reader = csv.reader(
                line for line in lines
                if line.strip() and not line.startswith("#")
            )

            for row in reader:
                if len(row) < 3:
                    continue

                try:
                    hostname = urlparse(row[2]).hostname
                except ValueError:
                    continue

                if not hostname:
                    continue

                hostname = hostname.lower().rstrip(".")

                if not is_ip_address(hostname):
                    urlhaus_domains.add(hostname)

    tranco_domains = set()

    with zipfile.ZipFile(tranco_path, "r") as archive:
        with archive.open("top-1m.csv") as raw_file:
            reader = csv.reader(
                line.decode("utf-8", errors="replace")
                for line in raw_file
            )

            for row in reader:
                if len(row) < 2:
                    continue

                domain = row[1].strip().lower().rstrip(".")

                if domain:
                    tranco_domains.add(domain)

    tranco_registrable = {
        get_registrable_domain(domain)
        for domain in tranco_domains
    }
    tranco_registrable.discard("")

    filtered_urlhaus = {
        domain
        for domain in urlhaus_domains
        if get_registrable_domain(domain) not in tranco_registrable
    }

    sample_size = len(filtered_urlhaus)

    rng = random.Random(RANDOM_SEED)
    sampled_tranco = rng.sample(
        sorted(tranco_domains),
        sample_size,
    )

    rows = []

    for domain in sorted(filtered_urlhaus):
        rows.append((domain, 1, "urlhaus"))

    for domain in sampled_tranco:
        rows.append((domain, 0, "tranco"))

    rng.shuffle(rows)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["hostname", "label", "source"])
        writer.writerows(rows)

    print()
    print("Baseline dataset created")
    print(f"URLhaus samples: {len(filtered_urlhaus):,}")
    print(f"Tranco samples: {len(sampled_tranco):,}")
    print(f"Total samples: {len(rows):,}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    urlhaus_path = RAW_DATA_DIR / "urlhaus_full_20260923.csv.zip"
    tranco_path = RAW_DATA_DIR / "tranco_top_1m_20260923.csv.zip"
    output_path = PROCESSED_DATA_DIR / "domain_risk_baseline_v0_1.csv"

    analyse_urlhaus_zip(urlhaus_path)
    analyse_tranco_overlap(urlhaus_path, tranco_path)
    build_baseline_dataset(
        urlhaus_path,
        tranco_path,
        output_path,
    )
