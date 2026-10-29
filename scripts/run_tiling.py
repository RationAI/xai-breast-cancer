from kube_jobs import storage, submit_job


submit_job(
    job_name="project_name-tiling",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    memory="32Gi",  # approximately 4GiB per process
    public=False,
    script=[
        "git clone https://github.com/RationAI/project_name.git workdir",
        "cd workdir",
        "uv sync",
        "uv run -m preprocessing.tiling +experiment=preprocessing/<experiment_name>",
    ],
    storage=[storage.secure.DATA],
)
