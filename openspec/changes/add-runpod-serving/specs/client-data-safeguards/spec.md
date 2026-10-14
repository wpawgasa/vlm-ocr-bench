# Spec Delta

## Purpose

States the conditions under which client bank statement pages may be sent to a Runpod endpoint, and makes the worker image, the deployment tool, the client and a recorded human review enforce them.

## ADDED Requirements

### Requirement: No request or response content in worker logs
A worker SHALL NOT write request content (images, prompts) or response content (model text, tokens) to standard output, standard error or any log. The rendered serving command SHALL always disable request logging and output logging and cap logged prompt length at zero, and the deploy SHALL refuse a command that lacks these flags. The PaddleOCR-VL front SHALL log only method, path, status, payload size and timing.

#### Scenario: Sentinel absent from logs
- **WHEN** a request whose prompt contains a unique sentinel string is sent to a worker and the worker's logs are collected afterwards
- **THEN** the logs contain the request's status and timing but not the sentinel and not the response text

#### Scenario: Logging flag missing
- **WHEN** a configuration's extra arguments re-enable request logging
- **THEN** `deploy` exits non-zero naming the flag and deploys nothing

### Requirement: Images only in memory, nothing persisted
A worker SHALL keep page images and model outputs in memory only. The worker SHALL NOT attach a network volume, SHALL run with the model hub offline, SHALL set the multimodal processor cache to zero, and SHALL NOT write request payloads to disk. The only writable paths SHALL be temporary directories that die with the worker.

#### Scenario: No volume on a promoted endpoint
- **WHEN** a configuration for a promotable endpoint names a network volume
- **THEN** `deploy` refuses it naming the volume

#### Scenario: Pipeline keeps the page in memory
- **WHEN** a `/layout-parsing` request is handled by the PaddleOCR-VL worker
- **THEN** no file containing the page bytes exists on the worker's filesystem after the response

### Requirement: Platform stores no payloads
Endpoints that may receive client data SHALL be load-balancing endpoints, so that the platform neither queues nor retains request or result payloads. If a queue-based endpoint is ever used for client data, its job time-to-live SHALL be set to the platform minimum, and the retention SHALL be stated in the review document before promotion.

#### Scenario: Queue endpoint not promotable
- **WHEN** `deploy <model> --promote` runs on a configuration with `endpoint.kind: queue` and the review document does not state the retention
- **THEN** the command exits non-zero naming the endpoint kind

### Requirement: Transport security
Every request to a Runpod endpoint SHALL use HTTPS. The client SHALL refuse a base URL for a Runpod host that uses plaintext HTTP before sending any request.

#### Scenario: Plaintext URL refused
- **WHEN** `DOTSOCR_BASE_URL` is `http://<id>.api.runpod.ai/v1`
- **THEN** building the model client raises an error naming the variable, and no request is sent

### Requirement: Keys restricted and never committed
Each endpoint SHALL be called with a Runpod API key restricted to that endpoint. The management key used to deploy SHALL be a separate key. Keys and endpoint ids SHALL exist only in the environment, the gitignored state file and the CI secret store. The repository SHALL run a secret scan in pre-commit and CI that fails on Runpod key and endpoint-host patterns.

#### Scenario: Key pattern in a commit
- **WHEN** a staged file contains a string matching the Runpod API key pattern
- **THEN** the pre-commit hook rejects the commit naming the file

#### Scenario: Unauthenticated request rejected
- **WHEN** a request is sent to a promoted endpoint without a key
- **THEN** the gateway rejects it with 401 or 403 and the smoke check records that the check passed

### Requirement: Region and tenancy pinning
Every endpoint SHALL be deployed only to data centres listed in the committed allow-list `configs/runpod/data_centers.yaml`, which SHALL contain only Secure Cloud data centres approved in the review document. A configuration with an empty data-centre list or one outside the allow-list SHALL be refused.

#### Scenario: Data centre outside allow-list
- **WHEN** a configuration lists a data centre id absent from the allow-list
- **THEN** `deploy` exits non-zero naming the id

#### Scenario: Allow-list change requires review
- **WHEN** the allow-list file changes
- **THEN** its test fails unless the review document's date is on or after the change and lists the new id

### Requirement: Human review of data-processing terms
Before any endpoint is promoted, a person SHALL review Runpod's data processing agreement, terms of service, compliance documents and region statements, and SHALL record in `docs/runpod-data-processing-review.md` the date, reviewer, documents reviewed, decision (`approved` or `rejected`), conditions, and the next review date at most 12 months later. The deploy tool SHALL refuse promotion while the decision is not `approved` or the review is older than 12 months. The review SHALL contain no client content, endpoint ids or keys.

#### Scenario: Promotion before review
- **WHEN** `deploy dotsocr --promote` runs and the review document is missing or its decision is not `approved`
- **THEN** the command exits non-zero naming the document

#### Scenario: Review expired
- **WHEN** the review's date is more than 12 months before today
- **THEN** promotion is refused and existing promotions are reported as expired by `status`

### Requirement: Audit of endpoints allowed to receive client data
The system SHALL keep, per endpoint, whether it may receive client data, who approved it and when, in the gitignored state file, and SHALL mirror the promotion (model name, date, parity verdict, approver) in the committed release log without ids. A client SHALL refuse to send a page from a client-data source to an endpoint that is not promoted. `ocrbench runpod status` SHALL list every endpoint with its promotion state.

#### Scenario: Statement page to an unpromoted endpoint
- **WHEN** `ocrbench infer` selects a `bankstmt` row and the model's base URL is a Runpod endpoint whose state is not promoted
- **THEN** the command exits non-zero before sending any page, naming the model and the endpoint name

#### Scenario: Public page to an unpromoted endpoint
- **WHEN** the parity or smoke check sends ThaiOCRBench pages to an unpromoted endpoint
- **THEN** the requests are allowed

#### Scenario: Audit listing
- **WHEN** `ocrbench runpod status` runs
- **THEN** every configured model is listed with its endpoint name, masked id, release, parity verdict, promotion state, approver and date
