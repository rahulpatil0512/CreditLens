# Authoritative sources to download for real grounding

The prototype ships with a SIMULATED internal policy and prepared context notes
so it runs offline. For a genuinely regulation-grounded demo, download the real
documents below and place the PDFs in this folder, then re-run RAG ingestion:

    python scripts/ingest_rag.py

The ingestion script reads every .md, .txt and .pdf in docs/policy/.

Suggested authoritative documents (verify the latest version yourself):

- EBA Guidelines on loan origination and monitoring (EBA/GL/2020/06)
    https://www.eba.europa.eu/  (search: "loan origination and monitoring")

- EBA Guidelines on PD estimation, LGD estimation and treatment of defaulted
  exposures (EBA/GL/2017/16)
    https://www.eba.europa.eu/  (search: "PD estimation LGD estimation")

- ECB Guide to internal models
    https://www.bankingsupervision.europa.eu/  (search: "guide to internal models")

- BaFin — supervisory publications
    https://www.bafin.de/EN/

- Basel Committee on Banking Supervision — credit risk standards
    https://www.bis.org/bcbs/

NOTE: This project does not redistribute these documents. Only the titles and
official starting points are listed here; download them from the official sites.
