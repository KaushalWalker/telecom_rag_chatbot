"""Load and validate the persisted ChromaDB used by the chatbot."""

import os
import shutil
import sqlite3
import stat
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import urlparse
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAMES = ("faq", "tickets", "guides")
INDEX_FILES = ("data_level0.bin", "header.bin", "length.bin")


class ChromaDatabaseError(RuntimeError):
    """Raised when the persisted ChromaDB cannot be loaded safely."""


def _has_valid_chroma_db(directory: Path) -> bool:
    """Check SQLite integrity, expected collections, vectors, and index files."""
    sqlite_path = directory / "chroma.sqlite3"
    if not directory.is_dir() or not sqlite_path.is_file():
        return False

    try:
        database_uri = f"{sqlite_path.resolve().as_uri()}?mode=ro"
        connection = sqlite3.connect(database_uri, uri=True)
        try:
            if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
                return False

            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            if not {"collections", "segments", "embeddings"}.issubset(tables):
                return False

            counts = dict(
                connection.execute(
                    """
                    SELECT c.name, COUNT(e.id)
                    FROM collections AS c
                    JOIN segments AS s ON s.collection = c.id
                    JOIN embeddings AS e ON e.segment_id = s.id
                    WHERE c.name IN (?, ?, ?)
                    GROUP BY c.name
                    """,
                    COLLECTION_NAMES,
                ).fetchall()
            )
            if any(counts.get(name, 0) < 1 for name in COLLECTION_NAMES):
                return False

            segment_ids = [
                row[0]
                for row in connection.execute(
                    """
                    SELECT s.id
                    FROM segments AS s
                    JOIN collections AS c ON c.id = s.collection
                    WHERE s.scope = 'VECTOR' AND c.name IN (?, ?, ?)
                    """,
                    COLLECTION_NAMES,
                )
            ]
        finally:
            connection.close()
    except (OSError, sqlite3.Error, ValueError):
        return False

    if not segment_ids:
        return False

    for segment_id in segment_ids:
        try:
            uuid.UUID(segment_id)
        except (ValueError, AttributeError):
            return False

        index_directory = directory / segment_id
        if not index_directory.is_dir() or any(
            not (index_directory / filename).is_file() for filename in INDEX_FILES
        ):
            return False

    return True


def _safe_extract(archive_path: Path, destination: Path) -> None:
    """Validate and extract the flat Chroma persistence archive safely."""
    try:
        with zipfile.ZipFile(archive_path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ChromaDatabaseError(
                    f"The ChromaDB ZIP failed its CRC check at {bad_member!r}."
                )

            entries = []
            seen_paths = set()
            segment_directories = set()
            has_database = False
            root = destination.resolve()

            for info in archive.infolist():
                raw_name = info.filename.replace("\\", "/")
                is_directory = info.is_dir() or raw_name.endswith("/")
                normalized_name = raw_name[:-1] if is_directory else raw_name
                windows_path = PureWindowsPath(normalized_name)
                parts = PurePosixPath(normalized_name).parts

                if (
                    not normalized_name
                    or "\x00" in normalized_name
                    or normalized_name.startswith("/")
                    or windows_path.drive
                    or windows_path.is_absolute()
                    or any(part in {"", ".", ".."} for part in normalized_name.split("/"))
                ):
                    raise ChromaDatabaseError(
                        f"The ChromaDB ZIP contains an unsafe path: {info.filename!r}."
                    )

                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ChromaDatabaseError(
                        f"The ChromaDB ZIP contains a symbolic link: {info.filename!r}."
                    )

                if len(parts) == 1 and parts[0] == "chroma.sqlite3" and not is_directory:
                    has_database = True
                elif len(parts) == 1 and is_directory:
                    try:
                        uuid.UUID(parts[0])
                    except ValueError as exc:
                        raise ChromaDatabaseError(
                            "The ChromaDB ZIP has an unexpected top-level directory."
                        ) from exc
                    segment_directories.add(parts[0])
                elif len(parts) == 2 and not is_directory:
                    try:
                        uuid.UUID(parts[0])
                    except ValueError as exc:
                        raise ChromaDatabaseError(
                            "The ChromaDB ZIP must contain its files directly under UUID directories."
                        ) from exc
                    segment_directories.add(parts[0])
                else:
                    raise ChromaDatabaseError(
                        "The ChromaDB ZIP must contain chroma.sqlite3 and UUID index directories "
                        "at its root, without an extra enclosing folder."
                    )

                relative_path = PurePosixPath(*parts)
                if relative_path.as_posix().casefold() in seen_paths:
                    raise ChromaDatabaseError(
                        f"The ChromaDB ZIP contains a duplicate path: {info.filename!r}."
                    )
                seen_paths.add(relative_path.as_posix().casefold())

                target = (root / Path(*parts)).resolve()
                try:
                    target.relative_to(root)
                except ValueError as exc:
                    raise ChromaDatabaseError(
                        f"The ChromaDB ZIP contains an unsafe path: {info.filename!r}."
                    ) from exc
                entries.append((info, target, is_directory))

            if not has_database or not segment_directories:
                raise ChromaDatabaseError(
                    "The ChromaDB ZIP is missing chroma.sqlite3 or its UUID index directories at the root."
                )

            destination.mkdir(parents=True, exist_ok=True)
            for info, target, is_directory in entries:
                if is_directory:
                    target.mkdir(parents=True, exist_ok=True)
                    continue

                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
    except zipfile.BadZipFile as exc:
        raise ChromaDatabaseError("The downloaded ChromaDB file is not a valid ZIP archive.") from exc


def _download_archive(release_url: str, archive_path: Path) -> None:
    parsed_url = urlparse(release_url)
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise ChromaDatabaseError(
            "CHROMA_DB_RELEASE_URL must be an HTTPS GitHub Release asset URL."
        )

    try:
        with urlopen(release_url, timeout=60) as response, archive_path.open("wb") as output:
            if urlparse(response.geturl()).scheme != "https":
                raise ChromaDatabaseError("The ChromaDB download redirected away from HTTPS.")
            shutil.copyfileobj(response, output)
    except ChromaDatabaseError:
        raise
    except Exception as exc:
        raise ChromaDatabaseError(
            "Could not download the ChromaDB release asset. Check CHROMA_DB_RELEASE_URL "
            "and confirm the asset is publicly reachable."
        ) from exc


def ensure_chroma_db() -> Path:
    """Return the valid persistent directory, downloading it only when needed."""
    if _has_valid_chroma_db(CHROMA_DIR):
        return CHROMA_DIR

    release_url = os.environ.get("CHROMA_DB_RELEASE_URL", "").strip()
    if not release_url:
        state = "missing" if not CHROMA_DIR.exists() else "invalid"
        raise ChromaDatabaseError(
            f"The ChromaDB database is {state} at {CHROMA_DIR}. Set CHROMA_DB_RELEASE_URL "
            "to the HTTPS URL of the chroma-db.zip GitHub Release asset."
        )

    try:
        CHROMA_DIR.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".chroma-load-", dir=CHROMA_DIR.parent
        ) as temporary_directory:
            temporary_directory = Path(temporary_directory)
            archive_path = temporary_directory / "chroma-db.zip"
            staged_directory = temporary_directory / "extracted"

            _download_archive(release_url, archive_path)
            _safe_extract(archive_path, staged_directory)
            if not _has_valid_chroma_db(staged_directory):
                raise ChromaDatabaseError(
                    "The downloaded ChromaDB archive is incomplete or does not contain "
                    "non-empty faq, tickets, and guides collections."
                )

            backup_directory = temporary_directory / "previous-database"
            has_previous_directory = CHROMA_DIR.exists() or CHROMA_DIR.is_symlink()
            if has_previous_directory:
                os.replace(CHROMA_DIR, backup_directory)
            try:
                os.replace(staged_directory, CHROMA_DIR)
            except Exception:
                if has_previous_directory:
                    os.replace(backup_directory, CHROMA_DIR)
                raise
    except ChromaDatabaseError:
        raise
    except Exception as exc:
        raise ChromaDatabaseError(
            "Failed to install a valid ChromaDB database from the release asset configured "
            "by CHROMA_DB_RELEASE_URL."
        ) from exc

    return CHROMA_DIR
