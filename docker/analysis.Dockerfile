FROM python:3.11-slim
# Analysis worker image. Pin a digest at runtime; this file is a build recipe, not a live eval claim.
RUN pip install --no-cache-dir "numpy" "pandas" "anndata>=0.10,<0.14" "scipy" "pyarrow" "h5py"
WORKDIR /workspace
