from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import hydra
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.ray import read_slides
from ratiopath.tiling import grid_tiles
from ratiopath.tiling.utils import row_hash


def tile(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "slide_id": row["id"],
            "x": x,
            "y": y,
        }
        for x, y in grid_tiles(
            slide_extent=(row["extent_x"], row["extent_y"]),
            tile_extent=(row["tile_extent_x"], row["tile_extent_y"]),
            stride=(row["stride_x"], row["stride_y"]),
            last="keep",
        )
    ]


@with_cli_args(["+preprocessing=tiling"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides_path: list[str] = ...  # Load slide paths

    # One row per slide with its metadata and a unique `id`, see the ratiopath quick
    # start: https://rationai.github.io/ratiopath/learn/get-started/quick-start/tiling/
    slides = read_slides(
        slides_path, mpp=config.mpp, tile_extent=config.tile_size, stride=config.stride
    ).map(row_hash, num_cpus=0.1, memory=128 * 1024**2)

    tiles = slides.flat_map(tile, num_cpus=0.2, memory=128 * 1024**2)
    # Project-specific: keep only the relevant tiles, e.g. by their overlap with the
    # tissue mask (ratiopath example "Build A Mask-First Tile Filtering Pipeline").
    tiles = ...

    with TemporaryDirectory() as output_dir:
        slides.write_parquet(Path(output_dir, "slides").as_posix())
        tiles.write_parquet(Path(output_dir, "tiles").as_posix())

        logger.log_artifacts(
            local_dir=output_dir, artifact_path=config.mlflow_artifact_path
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
