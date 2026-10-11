# Independent trajectory evidence-index audit

Source run: `evotau-open-rc-pair-acquisition-airline-e10-v20-g1-c1-p2-20261010-readfix2`, frozen `0ae3f83`. Original artifacts and uncertain verdicts remain unchanged.

All four affected saved validator inputs have exactly the same message role sequence as their native trajectories; no projection shift explains these errors in this run. The Reviewer receives unnumbered lists and outputs incorrect positions:

| Task | Returned indices | Invalid evidence |
|---|---|---|
| 11 | 1, 3, 17 | 17 out of range; original Customer indices include 1,3,15 |
| 27 | 1,3,7,28,30 | 28 and 30 are assistant messages |
| 12 | 1,12,14,18 | 12/14 assistant; 18 out of range |
| 7 | 1,6,11 | 6 assistant; 11 tool |

This is a model citation-contract failure, not proof of four valid Customers. In particular Task 7's raw verdict says invalid, but its supporting citation is malformed; do not convert it into valid or restore old fitness eligibility.

Independent opt-in fix: `customer_evolution.trajectory_validator_version: v2`. Build an indexed view directly from saved native messages, retain original positions without flattening multi-tool wrappers, display message hashes and the exact allowed Customer indices. A separate v2 validator prompt/name asks for these indices. Existing strict schema, role checks and uncertainty handling are preserved. V1 defaults remain byte/behavior compatible. A/B configs both use the same v2 validator; it is not a Prompt V2 effect.

Tests exercise multi-tool projection divergence, exact raw hashes, invalid/assistant/tool/out-of-range/duplicate/bool citations, corrupted mapping and G2 resume without repeated stages. Input-side explicit indices may reduce live citation mistakes but cannot guarantee future model adherence. No model output is remapped, repaired or reclassified. No API requests were issued.
