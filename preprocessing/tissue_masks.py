from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import hydra
import pyvips
import ray
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.masks import tissue_mask, write_big_tiff
from ratiopath.openslide import OpenSlide


def process_slide(row: dict[str, Any], level: int, output_path: Path) -> dict[str, Any]:
    slide_path = Path(row["path"])
    with OpenSlide(slide_path) as slide:
        mpp_x, mpp_y = slide.slide_resolution(level)

    image = cast("pyvips.Image", pyvips.Image.new_from_file(slide_path, level=level))
    mask = tissue_mask(image, mpp=(mpp_x, mpp_y))
    mask_path = output_path / slide_path.with_suffix(".tiff").name

    write_big_tiff(mask, path=mask_path, mpp_x=mpp_x, mpp_y=mpp_y)
    return row


@with_cli_args(["+preprocessing=tissue_masks"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides_path: list[Path] = ...  # Load slide paths

    with TemporaryDirectory() as output_dir:
        ray.data.from_items([{"path": str(path)} for path in slides_path]).map(
            process_slide,
            fn_kwargs={"level": config.level, "output_path": Path(output_dir)},
            memory=3 * 1024**3,
        ).materialize()

        logger.log_artifacts(
            local_dir=output_dir, artifact_path=config.mlflow_artifact_path
        )


if __name__ == "__main__":
    main()
