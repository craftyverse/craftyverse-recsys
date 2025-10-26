"""File-system utilities."""
from __future__ import annotations
import pandas as pd
from pathlib import Path
from zipfile import ZipFile



def unzip_file(
    zip_path: str | Path,
    destination: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Extract a zip archive into a destination directory.

    Args:
        zip_path: Path to the .zip file that should be extracted.
        destination: Directory to extract into. Defaults to the zip's parent directory.
        overwrite: When False, raise FileExistsError if extraction would clobber files.

    Returns:
        Path to the directory that now contains the extracted archive.

    Raises:
        FileNotFoundError: zip_path does not exist.
        FileExistsError: overwrite is False and destination already has extracted files.
    """

    archive = Path(zip_path).expanduser()
    if not archive.exists():
        raise FileNotFoundError(f"Zip file not found: {archive}")

    target_dir = Path(destination).expanduser() if destination else archive.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(archive) as zip_file:
        if not overwrite:
            conflicts = [
                name
                for name in zip_file.namelist()
                if name.rstrip("/") and (target_dir / name).exists()
            ]
            if conflicts:
                raise FileExistsError(
                    "Destination already contains files from archive. "
                    "Pass overwrite=True to skip this check."
                )

        zip_file.extractall(target_dir)

    return target_dir


def read_csv(file_path: str | Path | None = None) -> pd.DataFrame:
    """
    Read a CSV file into a pandas DataFrame.

    Args:
        file_path: Path to the CSV file.

    Returns:
        A pandas DataFrame containing the data from the CSV file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        pd.errors.EmptyDataError: If the file is empty.
        pd.errors.ParserError: If there is a parsing error while reading the CSV.
    """
    if file_path is None:
        raise ValueError("file_path must be provided")

    path = Path(file_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    return df
