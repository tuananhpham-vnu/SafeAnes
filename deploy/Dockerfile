# SafeAnes ward demo: UI + API in one image. Build from the repo root:
#   docker build -f deploy/Dockerfile -t safeanes-ward .
# Run (bundle of export_ward.py mounted, or downloaded at start from a private HF dataset):
#   docker run -p 8000:8000 -v "<repo>/deploy/backend/data_ward:/srv/data_ward:ro" safeanes-ward
#   docker run -p 8000:8000 -e SAFEANES_BUNDLE_REPO=<user>/<dataset> -e HF_TOKEN=<read token> safeanes-ward
# The image holds no patient data and no model files: both are in the bundle.

# ---------- UI
FROM node:20-slim AS ui
WORKDIR /ui
COPY deploy/frontend/package.json deploy/frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY deploy/frontend/index.html deploy/frontend/tsconfig.json deploy/frontend/vite.config.ts ./
COPY deploy/frontend/src ./src
RUN npm run build

# ---------- API (Python version and library versions as on the Kaggle image the models were trained on:
# the TabM model and its scikit-learn preprocessing are pickles)
FROM python:3.13-slim
ARG TORCH_VERSION=2.11.0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY deploy/backend/requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}" \
 && pip install -r requirements.txt
# uc04.tabm: the class of the pickled TabM model (nothing else of the training package is needed)
COPY src/uc04/__init__.py src/uc04/tabm.py ./uc04/
COPY deploy/backend/app ./app
COPY deploy/backend/fetch_bundle.py .
COPY --from=ui /ui/dist ./ui
RUN useradd --create-home --uid 10001 app && mkdir -p /srv/data_ward && chown app /srv/data_ward
USER app
ENV SAFEANES_DATA=/srv/data_ward SAFEANES_UI=/srv/ui SAFEANES_ENGINE=live PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health', timeout=4)"
# one worker: the ward caches (probabilities, alarm states) are shared by every viewer of the process
CMD ["sh", "-c", "python fetch_bundle.py && exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers"]
