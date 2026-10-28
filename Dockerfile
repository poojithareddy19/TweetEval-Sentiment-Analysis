# Streamlit app with all four committed models, for local use.
# Run `git lfs pull` before building: the build fails if any model file is an LFS pointer.
#
#   docker build -t tweeteval-sentiment .
#   docker run --rm -p 8501:8501 tweeteval-sentiment
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# torch comes from the PyTorch CPU index (same version pin as requirements/inference.txt).
# The default PyPI wheel on Linux pulls several GB of CUDA libraries the app does not need.
COPY requirements/inference.txt requirements/inference.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu "$(grep '^torch==' requirements/inference.txt | tr -d '\r')" \
    && pip install -r requirements/inference.txt

COPY . .
RUN python scripts/check_lfs_pointers.py

# The app only reads /app, so root ownership is fine; a chown here would duplicate the 550 MB of models.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).read() != b'ok')"

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address", "0.0.0.0", "--server.headless", "true", "--browser.gatherUsageStats", "false"]
