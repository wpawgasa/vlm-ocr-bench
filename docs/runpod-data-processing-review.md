---
date: 2026-10-19
reviewer: Wichai P.
documents:
  - Data owner's confirmation (2026-10-19) that the bank statements in this repository are approved for experimental use, including processing by an outside cloud provider (Runpod) and outside Thailand
decision: approved
conditions:
  - Experimental scope only. The bank statements already in this repository and the public ThaiOCRBench pages; no production client traffic.
  - Data centres limited to configs/runpod/data_centers.yaml.
  - Every other design D9 safeguard applies. Workers log no request content, client-data endpoints have no network volume, an endpoint is promoted only after a parity pass, and promotions are recorded in .runpod/state.yaml and configs/runpod/releases.yaml.
  - Before any production use, the full review in #139 replaces this document. It covers Runpod's DPA, Terms of Service, Trust Center (SOC 2 Type II scope), the region statement, worker-log retention in writing, and Thai PDPA cross-border transfer.
next_review: 2027-01-19
---

# Runpod data-processing review (experimental scope)

## Decision

Runpod endpoints may process the bank statements in this repository for experiments, in
the data centres on the allow-list. The data owner has confirmed that these statements may
be sent to an outside cloud provider and processed outside Thailand for experimental use.

This is **not** the production review. Runpod's contractual and compliance documents have
not been reviewed yet; that is deferred to #139 (task 5.3), which must be done and this file
replaced before any production client traffic.

## Data-centre allow-list

`configs/runpod/data_centers.yaml` holds every Secure Cloud data centre that, on 2026-10-19,
listed at least one of the GPUs the endpoint configs request (L4, RTX A5000, RTX 3090):
EU-RO-1, EU-SE-1, EU-CZ-1, EUR-IS-1, EUR-IS-2, CA-MTL-1, US-CA-2, US-GA-2, US-IL-1, US-MO-2.
Stock of these GPUs was low everywhere, so the list is broad to reduce "no worker
available" stalls. AP-JP-1 and AP-IN-1 had none of these GPUs. Round-trip latency from
Thailand is about 200–250 ms to either Europe or the US, which is small next to a 5–20 s page.

## What protects the data in the meantime

These are the design D9 safeguards, enforced in code:

- Workers run vLLM with request logging off and `--max-log-len 0`, and `deploy` refuses an
  argv without those flags. The Paddle front logs only status, size and timing.
- Load-balancing endpoints don't store payloads on the platform. Client-data endpoints have
  no network volume, and images stay in memory.
- The client refuses plain `http://` for `*.api.runpod.ai`.
- Bank-statement pages go only to an endpoint promoted with `deploy --promote`. Promotion
  needs a parity pass and this review to be `approved` and less than 12 months old.
- Keys live only in the environment, the gitignored `.env` and `.runpod/`, and GitHub
  repository secrets.

## Open for the production review (#139)

- Runpod's DPA, Terms of Service, and Trust Center reports (SOC 2 Type II scope: Secure Cloud).
- Runpod's written answer on serverless worker-log and gateway-log retention and staff access.
- Thai PDPA requirements for cross-border transfer, signed off by legal or compliance.
- Whether to narrow the allow-list (for example, EU only for GDPR coverage).
