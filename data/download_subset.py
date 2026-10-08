from __future__ import annotations

import argparse
import os
import sys
import tarfile
import urllib.request
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

# This puts the downloaded files in:
# data/raw/
#
# because this script is located at:
# data/download_subset.py
RAW_DIR = Path(__file__).resolve().parent / "raw"


# ============================================================
# Download function
# ============================================================

def download(n: int) -> None:

    # --------------------------------------------------------
    # Get FASTMRI URL from environment variable
    # --------------------------------------------------------

    url =  "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_singlecoil_val.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=IaE17GH6gzIbzh5xFHJ7s7qGKKA%3D&Expires=1796650695"

    if not url:
        sys.exit(
            "ERROR: FASTMRI_VAL_URL is not set.\n\n"
            "Set it first with:\n\n"
            'export FASTMRI_VAL_URL="YOUR_FASTMRI_URL"\n'
        )

    # --------------------------------------------------------
    # Validate number of volumes
    # --------------------------------------------------------

    if n <= 0:
        sys.exit("ERROR: --n must be greater than 0.")

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    RAW_DIR.mkdir(parents=True, exist_ok=True)


    print("FASTMRI Single-Coil Validation Dataset")

    print(f"Number of volumes requested: {n}")
    print(f"Output directory: {RAW_DIR}")
    print()
    print("Starting download...")
    print("The archive will be streamed.")
    print("Only the requested number of .h5 volumes will be extracted.")
    print()

    kept = 0

    try:

        # ----------------------------------------------------
        # Open FASTMRI URL
        # ----------------------------------------------------

        with urllib.request.urlopen(url) as response:

            # ------------------------------------------------
            # Open compressed TAR archive in streaming mode
            # ------------------------------------------------

            with tarfile.open(
                fileobj=response,
                mode="r|*"
            ) as tar:

                # --------------------------------------------
                # Read archive one file at a time
                # --------------------------------------------

                for member in tar:

                    # We only want HDF5 files
                    if not member.name.lower().endswith(".h5"):
                        continue

                    # ----------------------------------------
                    # Get only the filename
                    # ----------------------------------------

                    filename = Path(member.name).name

                    # Ignore strange/empty filenames
                    if not filename:
                        continue

                    output_file = RAW_DIR / filename

                    # ----------------------------------------
                    # Check if already downloaded
                    # ----------------------------------------

                    if output_file.exists():

                        print(
                            f"[{kept + 1}/{n}] "
                            f"Already exists: {filename}"
                        )

                        kept += 1

                    else:

                        # ------------------------------------
                        # SECURITY:
                        # Only extract files into RAW_DIR.
                        #
                        # We already flattened the filename
                        # using Path(...).name, so archive
                        # paths such as ../../something cannot
                        # escape the output directory.
                        # ------------------------------------

                        member.name = filename

                        # Compatible with older Python versions.
                        tar.extract(
                            member,
                            path=RAW_DIR
                        )

                        kept += 1

                        print(
                            f"[{kept}/{n}] "
                            f"Downloaded: {filename}"
                        )

                    # ----------------------------------------
                    # Stop after requested number
                    # ----------------------------------------

                    if kept >= n:
                        break


    except urllib.error.HTTPError as e:

        print()

        print("DOWNLOAD ERROR")

        print(f"HTTP error: {e.code}")
        print(f"Reason: {e.reason}")
        print()
        print(
            "Your FASTMRI signed URL may have expired."
        )
        print(
            "Get a fresh FASTMRI download URL and try again."
        )

        sys.exit(1)

    except urllib.error.URLError as e:

        print()
        print("=" * 60)
        print("NETWORK ERROR")
        print("=" * 60)
        print(f"Reason: {e.reason}")
        print()
        print(
            "Check your internet connection and try again."
        )

        sys.exit(1)

    except tarfile.TarError as e:

        print()
        print("=" * 60)
        print("ARCHIVE ERROR")
        print("=" * 60)
        print(e)

        sys.exit(1)

    except Exception as e:

        print()
        print("=" * 60)
        print("ERROR")
        print("=" * 60)
        print(e)

        sys.exit(1)

    # ========================================================
    # Finished
    # ========================================================

    print()
    print("=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    print(f"Volumes saved: {kept}")
    print(f"Location: {RAW_DIR}")

    print()
    print("Downloaded files:")

    for file in sorted(RAW_DIR.glob("*.h5")):
        print(f"  - {file.name}")

    print()

    if kept < n:
        print(
            f"WARNING: Requested {n} volumes, "
            f"but only found {kept} .h5 files."
        )
    else:
        print(f"Successfully downloaded {kept} volume(s).")

    print()
    print("Next step:")
    print("  python data/preprocess.py")


# ============================================================
# Command-line interface
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Download a subset of FASTMRI "
            "single-coil validation volumes."
        )
    )

    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of volumes to download (default: 5)"
    )

    args = parser.parse_args()

    download(args.n)


# ============================================================
# Run program
# ============================================================

if __name__ == "__main__":
    main()