# Maintainer policy

Production releases require two active human maintainers with repository write
access. Each release needs approval from both maintainers for:

- runtime and memory-safety changes;
- compatibility and wheel matrices;
- hardware CI evidence;
- fallback and numerical acceptance reports.

The repository has not supplied an authoritative maintainer roster in this
checkout, and the GitHub API does not expose collaborator membership to the
current contributor token. Names must be added here by the organization rather
than inferred from commit authorship.

Until two maintainers accept these responsibilities, the backend remains
alpha and does not meet the production-ready definition in Issue #4.
