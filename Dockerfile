# Container image for the MCP tool server (mcp_server.py) — see ACA_Deploy.md.
#
# Only the deployable subset goes in: 8 first-party modules plus agent_obs and
# the JSON stores. cma.py, gen_cma_yaml.py and manual_setup.py are client-side
# and stay out, along with tests/, var/ and .env (see .dockerignore).

FROM python:3.14-slim

# PYTHONUNBUFFERED so uvicorn's output reaches `az containerapp logs` as it
# happens rather than when a buffer happens to fill.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first: this layer is cached across source edits, so a code-only
# change rebuilds in seconds instead of reinstalling the wheel set.
COPY requirements-mcp.txt ./
RUN pip install --no-cache-dir -r requirements-mcp.txt

COPY agent_obs/ ./agent_obs/
COPY mcp_server.py repl.py roles.py tools.py storage.py \
     data_entities.py agent_schemas.py agent_memory.py prompts.py ./

# Only the five stores storage.py reads plus the agent's memory file. The
# synthetic-incidents-*.json fixtures are test data and are left behind.
COPY data/adjusters.json data/incidents.json data/insurers.json \
     data/policies.json data/escalations.json data/agent_memory.json ./data/

# A pristine copy the entrypoint can seed a mounted volume from. Without it, a
# CLAIMS_DATA_DIR pointing at an empty share makes every load return {} — an
# empty world that looks like a working server.
RUN cp -r ./data ./seed

# Plain COPY plus an explicit chmod below, not `COPY --chmod=755`: that flag is
# BuildKit-only, and `az acr build` runs on the classic Docker builder, which
# fails the step outright. Do not reintroduce it — a local `docker build` with
# BuildKit on will happily accept it and the break only shows up in Azure.
COPY entrypoint.sh ./

# The unprivileged account the server actually runs as. Deliberately no
# `USER app` here: the container starts as root and `entrypoint.sh` drops to
# this uid with `setpriv` after seeding. A mounted volume arrives owned by root,
# so an image that had already dropped privileges could not seed it and
# crash-looped on `cp: Permission denied`. Seeding needs root; serving does not.
RUN useradd --create-home --uid 10001 app \
    && chmod 755 /app/entrypoint.sh \
    && chown -R app:app /app/data

# 0.0.0.0, not McpConfig's 127.0.0.1 default: a loopback bind is unreachable
# from outside the container and the health probe would fail with no clue why.
ENV MCP_HOST=0.0.0.0 \
    MCP_PORT=8787 \
    OBS_VAR_DIR=/tmp/obs

EXPOSE 8787

# /healthz is exempt from BearerAuth, so this needs no token.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8787/healthz').read()"

ENTRYPOINT ["./entrypoint.sh"]
