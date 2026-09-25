# Data statement

No real patient data are distributed with this repository. `data/data.csv` contains sample records; all entries are fictional and were written solely for software testing.

## Required fields

Training and retrieval accept CSV or JSONL with these normalized fields:

| Field | Required | Description |
| --- | --- | --- |
| `record_id` | yes | Unique record identifier; `ehr_id` is accepted as an alias in JSONL |
| `patient_id` | recommended | Used to reject patient overlap across splits |
| `center` | recommended | Site identifier used for stratified evaluation |
| `split` | yes | `train`, `validation`, or `test` |
| `label_high_risk` or `label` | yes | Binary high-risk target |
| `query_text` | yes | Information available for the query case |
| `retrieval_text` | yes | Reference text indexed as the document |
| `secondary_endpoint` | conditional | Binary auxiliary target used by the study recipe |
| `diagnosis_teacher_row_id` | conditional | Row in the governed diagnosis prototype matrix |
| `metadata` | no | JSON object with non-identifying provenance fields |

Use `prepare_data.py` to validate identifiers and export normalized JSONL. The public sample CSV omits patient, center, diagnosis, and secondary-endpoint supervision because it only exercises the risk-retrieval path.

## Study-only artifacts

The study datasets, diagnosis prototype matrix, cohort manifests, patient-level predictions, and trained checkpoint are not included. Access must follow the applicable institutional review, data-use agreement, and participating-site approval process.

Before training, users must verify that:

1. direct and indirect identifiers are handled under the approved protocol;
2. a patient cannot occur in more than one split;
3. site- and time-based validation rules match the study protocol;
4. generated text, indexes, logs, prompts, and outputs are protected as sensitive derivatives; and
5. clinical content is not sent to an external API without explicit authorization.

Private data must remain outside Git. The repository `.gitignore` excludes common data and output paths, but users remain responsible for reviewing every staged file.
