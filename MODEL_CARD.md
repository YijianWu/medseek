# MedSeek model card

## Model summary

MedSeek uses a shared E5 dual encoder to represent query and reference clinical text. The compact public training implementation supports supervised contrastive risk learning, multi-positive batch InfoNCE, diagnosis prototype-soft alignment, secondary-endpoint classification and margin objectives, and inference-aligned balanced retrieval scoring.

The default demonstration uses the public `Yjian1998/medseek` inference checkpoint. The underlying `intfloat/multilingual-e5-large` model remains available as a base-model baseline.

## Inputs and outputs

Inputs are query text and reference text. Training can additionally consume a high-risk label, a secondary-endpoint label, and a row identifier into an authorized diagnosis prototype matrix. Retrieval returns positive and negative neighbors, a continuous similarity margin, and classification metrics when labels are available.

## Intended use

- Research on clinical-text retrieval and risk-recognition methods.
- Reproduction studies performed under appropriate data governance.
- Software validation with the bundled sample data or properly authorized data.

## Out-of-scope use

- Autonomous diagnosis, triage, treatment, or discharge decisions.
- Direct clinical deployment without independent validation and regulatory review.
- Processing identifiable clinical records through unapproved external services.

## Limitations

Performance may vary across institutions, languages, documentation practices, demographic groups, and prevalence settings. Retrieval can expose information from its corpus, so indexes and neighbors must be protected. The bundled fictional examples are deliberately small and easy; their metrics do not estimate clinical performance.
