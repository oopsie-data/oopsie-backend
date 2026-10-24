"""Converters from external datasets to Oopsie-Data, by the name ``configs/imports.yaml`` uses.

Each converter is a command-line ``main(argv)`` that writes ``<episode>.h5`` files
and their videos into ``--output-dir``; ``source_flag`` names its input argument.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from oopsie_backend.conversion import (
    convert_ar_aloha_data,
    convert_rlds_to_hdf5,
    convert_roboarena_data,
    convert_soar_data,
)


@dataclass(frozen=True)
class Converter:
    main: Callable[[list[str]], object]
    source_flag: str


CONVERTERS: dict[str, Converter] = {
    "ar-aloha": Converter(convert_ar_aloha_data.main, "--source"),
    "rlds": Converter(convert_rlds_to_hdf5.main, "--rlds-version-dir"),
    "roboarena": Converter(convert_roboarena_data.main, "--source"),
    "soar": Converter(convert_soar_data.main, "--source"),
}
