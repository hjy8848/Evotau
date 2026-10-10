"""E-only Customer search with a frozen native Service; no Service or H search."""

from .customer_evolution import (
    evaluate_candidate,
    generate_candidates,
    prepare_candidate,
)
from .customer_skills import PROTOCOL
from .evolution_archive import CustomerAttemptHistory
from .evolution_artifacts import EvolutionJournal
from .strategies import PromptStrategy
from .tau_provenance import sha256_json


def run_customer_diagnostic(
    *,
    tasks,
    task_ids,
    runner,
    providers,
    service,
    policy,
    generations,
    count,
    seed,
    concurrency,
    domain_policy,
    output,
    manifest_sha,
    initial_customer=None,
    checkpoint_path=None,
):
    import json
    from pathlib import Path

    from .alternating import (
        _accuracy,
        _context_episodes,
        _run_panel,
        _write_json_atomic,
    )
    from .evolution_context import build_customer_evidence
    from .records import EpisodeRecord

    if set(tasks) != set(task_ids):
        raise ValueError("Customer diagnostic may load only the frozen E panel")
    providers.configure_customer({"customer_evolution": policy})
    journal = EvolutionJournal(output, manifest_sha)
    history = CustomerAttemptHistory(max_history=policy["max_history"])
    customer = initial_customer or PromptStrategy("")
    journal.freeze(
        "customer-diagnostic-plan",
        {
            "task_ids": task_ids,
            "seed": seed,
            "policy": policy,
            "generations": generations,
            "count": count,
            "concurrency": concurrency,
            "initial_customer": customer.to_dict(),
            "frozen_service": service.to_dict(),
            "domain_policy": domain_policy,
        },
        lambda: {"customer_protocol": PROTOCOL},
    )
    checkpoint = (
        Path(checkpoint_path)
        if checkpoint_path
        else Path(output) / "customer-diagnostic-checkpoint.json"
    )
    if checkpoint.is_symlink():
        raise ValueError("unsafe Customer diagnostic checkpoint")
    saved = json.loads(checkpoint.read_text()) if checkpoint.exists() else None
    if saved is not None and (
        saved.get("manifest_sha256") != manifest_sha
        or saved.get("customer_protocol") != PROTOCOL
        or saved.get("sha256")
        != sha256_json({k: v for k, v in saved.items() if k != "sha256"})
    ):
        raise ValueError("Customer diagnostic checkpoint identity/digest mismatch")
    if (
        saved is not None
        and not (
            journal.root
            / f"customer-diagnostic-g{saved['completed_generation']}-complete.json"
        ).exists()
    ):
        raise ValueError("Customer diagnostic checkpoint refers to missing generation")
    documents = []

    def freeze(name, inputs, callback):
        _write_json_atomic(
            Path(output) / "customer-diagnostic-stage.json",
            {
                "stage": name,
                "manifest_sha256": manifest_sha,
                "customer_protocol": PROTOCOL,
            },
        )
        return journal.freeze(name, inputs, callback)

    def panel(g, label, c):
        inputs = {
            "customer": c.to_dict(),
            "service": service.to_dict(),
            "ids": task_ids,
            "seed": seed,
        }
        result = freeze(
            f"customer-diagnostic-g{g}-{label}",
            inputs,
            lambda: {
                "episodes": [
                    r.to_dict()
                    for r in _run_panel(
                        runner,
                        tasks=tasks,
                        task_ids=task_ids,
                        seed=seed,
                        customer=c,
                        service=service,
                        panel_name=f"customer-diagnostic-g{g}-{label}",
                        max_parallel_episodes=concurrency,
                    )
                ]
            },
        )
        return tuple(EpisodeRecord.from_dict(v) for v in result["episodes"])

    for g in range(generations):
        incumbent = panel(g, "incumbent", customer)
        context = {
            "customer_evolution": policy,
            "generation": g,
            "task_interactions": build_customer_evidence(
                _context_episodes(incumbent, runner, tasks),
                representative_cases=policy["representative_cases"],
                case_chars=policy["case_chars"],
                customer_protocol=PROTOCOL,
            ),
            "current_customer_strategy": customer.text,
            "current_service": service.to_dict(),
            "customer_attempt_history": history.entries,
            "service_policy": domain_policy,
        }
        proposed = freeze(
            f"customer-diagnostic-g{g}-proposals",
            context,
            lambda context=context: generate_candidates(providers, context, count),
        )
        selected_accuracy = _accuracy(incumbent)
        rows = []
        if proposed.get("generation_rejection"):
            rows.append(proposed["generation_rejection"])
        tested_procedures = {entry.get("procedure_id") for entry in history.entries}
        for i, proposal in enumerate(proposed["candidates"]):
            row, skill = prepare_candidate(proposal, context, policy)
            if skill is not None and (
                skill.compile().text == customer.text
                or skill.procedure_id in tested_procedures
            ):
                row.update(
                    candidate_validity="invalid",
                    pre_rollout_rejection="duplicate Customer procedure",
                )
                skill = None
            if skill is not None:
                tested_procedures.add(skill.procedure_id)
            row, _ = evaluate_candidate(
                row,
                skill,
                context=context,
                incumbent=incumbent,
                current_service=service,
                tasks=tasks,
                runner=runner,
                providers=providers,
                domain_policy=domain_policy,
                panel=lambda c, s, i=i, g=g: panel(g, f"candidate-{i}", c),
                stage=lambda n, inputs, cb, g=g, i=i: freeze(
                    f"customer-diagnostic-g{g}-c{i}-{n}", inputs, cb
                ),
            )
            if row["eligible_for_selection"] and row["accuracy"] < selected_accuracy:
                customer, selected_accuracy = skill.compile(), row["accuracy"]
            rows.append(row)
        for row in rows:
            row["selected"] = bool(
                row.get("eligible_for_selection")
                and row.get("strategy", {}).get("text") == customer.text
            )
            if row.get("eligible_for_selection"):
                row["selection_reason"] = (
                    "selected: strict native E accuracy decrease"
                    if row["selected"]
                    else "no strict native E accuracy decrease"
                    if row["accuracy"] >= _accuracy(incumbent)
                    else "another eligible candidate has lower native E accuracy"
                )
            history.add(row, g)
        doc = freeze(
            f"customer-diagnostic-g{g}-complete",
            {
                "rows": rows,
                "customer": customer.to_dict(),
                "service": service.to_dict(),
            },
            lambda rows=rows, selected_accuracy=selected_accuracy, g=g, incumbent=incumbent: {
                "generation": g,
                "incumbent_accuracy": _accuracy(incumbent),
                "selected_accuracy": selected_accuracy,
                "candidates": rows,
            },
        )
        documents.append(doc)
        state = {
            "customer_protocol": PROTOCOL,
            "manifest_sha256": manifest_sha,
            "completed_generation": g,
            "customer": customer.to_dict(),
            "frozen_service": service.to_dict(),
            "attempt_history": history.entries,
            "generation_sha256": sha256_json(doc),
        }
        state["sha256"] = sha256_json(state)
        if saved is not None and saved["completed_generation"] == g and saved != state:
            raise ValueError(
                "Customer diagnostic checkpoint differs from committed generation"
            )
        if saved is None or g > saved["completed_generation"]:
            _write_json_atomic(checkpoint, state)
    return {
        "status": "complete",
        "customer_protocol": PROTOCOL,
        "generations": documents,
        "customer": customer.to_dict(),
        "frozen_service": service.to_dict(),
        "attempt_history": history.entries,
        "validation_loaded": False,
        "heldout_loaded": False,
        "causal_service_failure_confirmed": False,
    }
