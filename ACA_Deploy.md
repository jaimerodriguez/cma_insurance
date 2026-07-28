# Deploying the MCP tool server to Azure Container Apps

This deploys **only** `mcp_server.py` and what it needs — the 31 role-restricted
domain tools, served over streamable HTTP. `repl.py`'s REPL and `cma.py` stay on
your machine; they are the *clients*.

**Why Container Apps and not Azure Functions.** Functions' MCP extension wants one
`mcpToolTrigger` function per tool with a flat `toolProperties` list that cannot
express enums, nested objects, or type unions — so it would discard
`agent_schemas.build_tool_schemas()` and silently regress the nullable-optional
fix. Its ASGI shim also buffers response bodies ("Chunked bodies not supported"),
which breaks SSE. Container Apps runs `mcp_server.py` unmodified, streams
properly, and still scales to zero. See the CHANGELOG entry for the full
comparison.

---

## 1. What you need

- An Azure subscription and the Azure CLI (`az --version` ≥ 2.60).
- `az extension add --name containerapp --upgrade`
- `az login`

**You do not need Docker locally.** `az acr build` builds the image in Azure from
this directory.

---

## 2. Set your variables

Every command below uses these. Paste this block into your shell first.

```bash
export RG=rg-claims-mcp
export LOC=eastus
export ACR=claimsmcp$RANDOM          # must be globally unique, lowercase alnum
export ENVNAME=cae-claims
export APP=claims-mcp
export IMAGE=claims-mcp:v1
```

Generate the bearer token with the script's own helper so the format matches
what `McpConfig` expects:

```bash
python3 mcp_server.py --new-token          # prints MCP_BEARER_TOKEN=...
export MCP_BEARER_TOKEN=<paste the value>
```

Keep this token. You need the identical value in your local `.env` in step 5.

---

## 3. One-time setup

```bash
az group create --name $RG --location $LOC

# Basic SKU is the cheapest that supports `az acr build`.
az acr create --resource-group $RG --name $ACR --sku Basic --admin-enabled true

# Build in the cloud from this directory. Honours .dockerignore.
az acr build --registry $ACR --image $IMAGE .

# --logs-destination none skips the Log Analytics workspace and its ingestion
# bill. Swap to `--logs-destination log-analytics` if you want queryable logs;
# `az containerapp logs show` still works either way for live tailing.
az containerapp env create \
  --name $ENVNAME --resource-group $RG --location $LOC \
  --logs-destination none
```

Then create the app:

```bash
ACR_PASS=$(az acr credential show -n $ACR --query "passwords[0].value" -o tsv)

az containerapp create \
  --name $APP --resource-group $RG --environment $ENVNAME \
  --image $ACR.azurecr.io/$IMAGE \
  --registry-server $ACR.azurecr.io \
  --registry-username $ACR --registry-password "$ACR_PASS" \
  --target-port 8787 --ingress external \
  --cpu 0.5 --memory 1.0Gi \
  --min-replicas 0 --max-replicas 1 \
  --secrets mcp-token="$MCP_BEARER_TOKEN" \
  --env-vars MCP_BEARER_TOKEN=secretref:mcp-token OBS_VAR_DIR=/tmp/obs
```

Three of those flags are load-bearing, not defaults worth skimming:

| Flag | Why |
|---|---|
| `--max-replicas 1` | `mcp_server._STORAGE_LOCK` is a `threading.Lock` — it serializes writes **within one process**. `storage.py` is whole-file read-modify-write, so a second replica would lose the first's writes across unrelated records. One replica is a correctness constraint here, not a cost choice. |
| `--min-replicas 0` | Scale to zero. This is what makes idle cost nothing. |
| `--secrets` + `secretref:` | Keeps the token out of the revision's plain-text env vars, where `az containerapp show` would print it. |

The app defaults to **single-revision mode**, which matters: in multi-revision
mode an update runs the old and new revisions concurrently and you'd briefly have
two writers. Don't switch it.

Grab the URL:

```bash
export FQDN=$(az containerapp show -n $APP -g $RG \
  --query properties.configuration.ingress.fqdn -o tsv)
echo "https://$FQDN"
```

---

## 4. Verify

```bash
# /healthz is exempt from auth by design — no token needed.
curl -s https://$FQDN/healthz | python3 -m json.tool
```

Expect `{"ok": true, "run_id": "...", "tools": {"adjuster": 21, "insurer": 8, "agent": 12}}`.
Those three counts are the real check: they mean `agent_schemas` ran and the
role tables built correctly inside the container.

Confirm the auth gate actually closes:

```bash
curl -s -o /dev/null -w '%{http_code}\n' https://$FQDN/mcp/agent     # 401
curl -s -o /dev/null -w '%{http_code}\n' \
  -H "Authorization: Bearer $MCP_BEARER_TOKEN" \
  -H 'Accept: application/json, text/event-stream' \
  -H 'Content-Type: application/json' \
  -X POST https://$FQDN/mcp/agent \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}'
                                                                      # 200
```

A `401` on the second command means the token in the container secret and the
one in your shell have drifted.

---

## 5. Point `cma.py` at it

Add to your local `.env`:

```
MCP_PUBLIC_URL=https://<paste $FQDN here>
MCP_BEARER_TOKEN=<the same token>
```

`McpConfig.active` auto-derives from having both, so this is the switch — no
code change. Confirm before running anything:

```bash
python3 mcp_server.py --print-config     # token is fingerprinted, not shown
```

Then `python3 cma.py` and `/mcp` to inspect, or `/agent` to run. `cma.update_agents`
detects the `mcp_servers` drift and repoints the stored agents; `cma.ensure_vault`
reconciles one `static_bearer` vault credential per endpoint URL.

To fall back to custom tools, set `MCP_TOOLS=0` (or clear `MCP_PUBLIC_URL`) and
re-run — `_desired_agent` sends `mcp_servers: []` and clears the declaration.

---

## 6. Start and stop — controlling cost

**The default already costs nothing when idle.** With `--min-replicas 0`, Container
Apps terminates the replica after ~5 minutes without requests and you are billed
for zero compute until the next request. For seldom use, you can stop reading
here — the standing cost is the registry, not the app.

Four levels, cheapest last:

### Level 0 — leave it alone (recommended)

Nothing to run. Idle costs nothing; the first request after idle pays a cold
start of roughly 10–20 s while Azure pulls and boots the container.

If a cold start inside a Managed Agents run makes you nervous, warm it first:

```bash
curl -s https://$FQDN/healthz > /dev/null      # then start your run
```

### Level 1 — keep it warm for a working session

Removes cold starts for the duration. **Bills continuously while set**, so put
the two commands around your session rather than leaving the first one on.

```bash
az containerapp update -n $APP -g $RG --min-replicas 1     # before
az containerapp update -n $APP -g $RG --min-replicas 0     # after — do not skip
```

### Level 2 — hard off

Level 0 already costs nothing, so this is for when you want the endpoint to stop
answering at all — no cold starts, no public URL:

```bash
az containerapp ingress disable -n $APP -g $RG            # off
az containerapp ingress enable  -n $APP -g $RG \
  --type external --target-port 8787 --transport auto     # back on
```

Re-enabling assigns the **same FQDN**, so your `.env` and the vault credential
stay valid.

### Level 3 — delete the app, keep the image

Drops the app entirely while leaving the registry and environment, so redeploying
is one command (step 3's `az containerapp create`, unchanged).

```bash
az containerapp delete -n $APP -g $RG --yes
```

The FQDN **changes** on recreate. Update `MCP_PUBLIC_URL` in `.env`;
`cma.ensure_vault` keys credentials by URL and will reconcile the new one.

### Level 4 — remove everything

```bash
az group delete --name $RG --yes --no-wait
```

---

## 7. What this actually costs

| Item | When idle | Notes |
|---|---|---|
| Container Apps compute | **$0** | Scale-to-zero. Azure grants 180,000 vCPU-s + 360,000 GiB-s + 2M requests free per subscription per month — at 0.5 vCPU / 1 GiB that's roughly **100 hours of runtime free every month**. Seldom use will not exceed it. |
| Container Apps environment | **$0** | Consumption-only environments have no baseline charge. |
| Log Analytics | **$0** | Avoided by `--logs-destination none`. |
| **Container Registry (Basic)** | **~$5/month** | The only standing cost, and it accrues whether or not the app runs. |

So the bill is dominated by the registry, not the compute. Two ways to avoid it:

- **Delete the ACR between uses** and re-run `az acr create` + `az acr build`
  when you next need it (a few minutes).
- **Use GitHub Container Registry** instead — free for public images, and within
  a generous quota for private ones. Push to `ghcr.io/<user>/claims-mcp:v1` and
  pass `--registry-server ghcr.io` with a PAT as the password.

Rates and free grants change; check the
[Container Apps pricing page](https://azure.microsoft.com/pricing/details/container-apps/)
before relying on the figures above.

---

## 8. Updating the image

```bash
az acr build --registry $ACR --image claims-mcp:v2 .
az containerapp update -n $APP -g $RG --image $ACR.azurecr.io/claims-mcp:v2
```

Use a new tag each time. Reusing `:v1` leaves the revision pointing at a digest
it already has, and the update becomes a no-op with no error.

---

## 9. State: read this before you rely on it

**In the default deployment above, data does not survive scale-to-zero.** Tool
writes land in the container's writable layer; when the replica terminates, the
world resets to the seed baked into the image. Within a single warm session
everything persists normally.

For a demo that is often what you want — every run starts from a known state. If
you need durability across runs, mount an Azure Files share:

```bash
export SA=claimsdata$RANDOM
az storage account create -n $SA -g $RG -l $LOC --sku Standard_LRS --kind StorageV2
az storage share-rm create -g $RG --storage-account $SA -n claims-data --quota 1

SA_KEY=$(az storage account keys list -g $RG -n $SA --query "[0].value" -o tsv)
az containerapp env storage set \
  --name $ENVNAME --resource-group $RG --storage-name claimsdata \
  --azure-file-account-name $SA --azure-file-account-key "$SA_KEY" \
  --azure-file-share-name claims-data --access-mode ReadWrite
```

Attaching the volume needs a YAML update — the CLI has no flag for it:

```bash
az containerapp show -n $APP -g $RG -o yaml > app.yaml
```

In `properties.template`, add the volume and its mount, and set `CLAIMS_DATA_DIR`:

```yaml
    containers:
    - name: claims-mcp
      env:
      - name: MCP_BEARER_TOKEN
        secretRef: mcp-token
      - name: OBS_VAR_DIR
        value: /tmp/obs
      - name: CLAIMS_DATA_DIR         # <-- add
        value: /data
      volumeMounts:                   # <-- add
      - volumeName: claims-data
        mountPath: /data
    volumes:                          # <-- add, sibling of `containers`
    - name: claims-data
      storageName: claimsdata
      storageType: AzureFile
```

```bash
az containerapp update -n $APP -g $RG --yaml app.yaml
```

`entrypoint.sh` copies the seed stores into an empty share on first boot, per
file, and never overwrites an existing one — a restart tops up a partially
populated volume rather than resetting a live world.

Two consequences worth knowing before you choose this:

- **Azure Files is SMB.** `storage.py` rewrites whole JSON files on every
  mutation, so each tool call becomes a network round trip to the share. Expect
  tens of milliseconds per call instead of 2–9 ms.
- **`--max-replicas 1` becomes load-bearing twice over.** A shared writable
  volume with two replicas and a process-local lock is silent data loss, not a
  race you'd notice.

`agent_obs` traces go to `/tmp/obs` and are lost with the replica either way. To
keep them, either add a second volume mount for `OBS_VAR_DIR` or leave
`--logs-destination log-analytics` on and read the stdout stream.

---

## 10. Troubleshooting

**Container exits immediately, logs show `MCP_BEARER_TOKEN is unset`.**
Working as designed — the server refuses to serve the claims tools
unauthenticated. The `secretref:` wiring is wrong; check
`az containerapp show -n $APP -g $RG --query properties.template.containers[0].env`.

**`/healthz` times out but the revision is "Running".**
Almost always a bind address. The image sets `MCP_HOST=0.0.0.0`; if you
overrode it, `McpConfig`'s `127.0.0.1` default is unreachable from outside the
container.

**Everything returns 401, including with the right token.**
Check for a trailing newline in the secret — `--secrets mcp-token="$(cat f)"`
keeps one, and `hmac.compare_digest` is exact.

**Tools return data, but it's an empty world (`{}` everywhere).**
`CLAIMS_DATA_DIR` points at a share the seeding step never reached. Confirm with
`az containerapp exec -n $APP -g $RG --command "ls -la /data"`.

**Live logs:**

```bash
az containerapp logs show -n $APP -g $RG --follow --tail 100
```

---

## 11. Security posture

This is the *most basic* authentication MCP supports, chosen deliberately:

- A single static bearer token, compared with `hmac.compare_digest`, over
  TLS terminated by Container Apps ingress.
- No per-caller identity, no expiry, no rotation, no rate limiting. Anyone with
  the token has the full tool surface for that role's endpoint.
- Role separation is enforced server-side by path (`/mcp/adjuster` cannot reach
  agent tools) — but one token covers all three endpoints.

To rotate: `az containerapp secret set -n $APP -g $RG --secrets mcp-token=<new>`,
then `az containerapp revision restart`, then update `.env`. The old token stops
working the moment the revision restarts, so update both together.

Before this holds anything real, the gaps to close are per-role tokens, expiry,
and pulling the secret from Key Vault via managed identity instead of ACR admin
credentials and a plain secret.
