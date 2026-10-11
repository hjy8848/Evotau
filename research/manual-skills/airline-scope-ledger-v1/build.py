"""Reproduce manually authored, E-evidence-only stimuli. No model calls."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import yaml

from evotau.alternating import _trajectory_context
from evotau.customer_skills import CustomerSkill
from evotau.evolution_candidates import validate_mutation
from evotau.service_skills import (
    ServiceSkillMemoryV2,
    apply_v2_mutation,
    validate_skill_budgets,
)
from evotau.tau_provenance import sha256_json

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]


def main():
    inventory = json.loads((PROJECT / 'research/audits/customer-evolution-root-cause-full-20261011/inventory.json').read_text())
    rows, refs = [], []
    for tid, run, panel, indices in [
        ('7', 'ab-v2-s1', 'candidate-1', [7, 12]),
        ('12', 'acquisition', 'incumbent', [7]),
        ('15', 'ab-v2-s1', 'incumbent', [15, 16, 17]),
    ]:
        e = next(e for e in inventory['episodes'] if e['task_id'] == tid and e['run'] == run and panel in e['panel'])
        native = json.loads(Path(e['path']).read_text())
        projected = _trajectory_context(native)['messages']
        messages = []
        for index in indices:
            ref = {'projected_message_index': index, 'message_sha256': sha256_json(projected[index])}
            messages.append({**projected[index], 'evidence_ref': ref})
            refs.append({'task_id': tid, 'seed': e['seed'], 'trajectory_ref': e['trajectory_ref'], **ref})
        rows.append({'task': {'task_id': tid}, 'seed': e['seed'], 'trajectory_ref': e['trajectory_ref'],
                     'original_native_path': e['path'], 'original_native_sha256': e['sha256'],
                     'native_evaluation': {'task_success': e['task_success'], 'reward': e['native_reward']},
                     'trajectory': {'messages': messages}})
    customer = {
        'schema_version': 1,
        'mechanism': 'Local consent across an interleaved, scenario-required information request.',
        'trigger': 'The original scenario has multiple pending operations or a conditional alternative, and a required side question interrupts discussion of one operation before all requested work is complete.',
        'procedure': [
            'Pursue exactly the original goals. Supply identities, booking facts, spending limits and payment constraints truthfully at the earliest time permitted by the scenario; preserve any explicit disclosure delay or question timing.',
            'When the scenario requires a side question, ask it at its prescribed time even if another requested action is pending. Ask only that original question; do not introduce another goal or tell the agent which tool or policy to use.',
            'When the agent quotes one option, consent only to that option if it satisfies the original conditions. Name the already-discussed booking and say that this confirmation covers that option only; do not give blanket consent for other pending changes.',
            'After the side question is answered, return concisely to the original pending requests. Keep each condition attached to its original booking or package. Answer a genuine clarification directly; do not repeat facts solely to teach the agent a solution.',
            'If the agent proposes an additional change not requested or a charge conflicting with a stated limit, point out the specific mismatch once and ask what action is actually proposed. Continue cooperating with every original valid request. Never withdraw a fixed goal. Never fabricate a fact. Never prolong the dialogue to cause failure.',
        ],
        'intensity': 'medium',
        'stop_conditions': ['All original goals and required information requests have been addressed.',
                            'A native human-transfer or terminal condition requires the conversation to end.',
                            'The procedure conflicts with a scenario constraint: immediately follow the original scenario instead.'],
        'hypothesis': 'An information detour and local confirmation can expose whether Service remembers operation-specific conditions rather than transferring one consent or budget between pending requests. It may also make consent clearer and improve accuracy; that outcome must be recorded rather than forced into a discovery claim.',
        'evidence_refs': refs[:3],
    }
    skill = CustomerSkill.from_mapping(customer, evidence_rows=rows)
    service = {
        'trigger': 'A write operation is pending while multiple user requests, a conditional option, a stated spending limit, or an intervening information question makes the exact consent scope easy to confuse.',
        'guidance': 'Maintain a compact internal per-operation ledger from observed conversation and tool results: affected booking, requested fields, conditions, quoted incremental cost and payment/refund method, and the exact scope of the latest consent. A side question does not erase pending goals or transfer their conditions. Immediately before a write, match its booking and changed fields to that ledger. A yes to one quoted option authorizes only that option, not another upgrade or every remaining request. Do not add an unrequested prerequisite when the requested action is already eligible. Check a stated cap against the complete requested package, including pending fees; do not silently treat a generic yes as revoking that cap. If a new quote conflicts with the cap or consent scope, ask one precise clarification before writing. State the affected booking, actual changes, cost and payment/refund consequences when requesting approval. Reuse an existing explicit approval when those details are unchanged; ask again only for a materially changed or ambiguous action. If conditions do not hold, explain that and offer only applicable alternatives without changing the user goal. Use only public policy, observed user statements and tool results; do not infer hidden budgets, invent policy exceptions, or expose this internal ledger. Resolve the original information question at its actual query time and then return to all pending goals.',
        'activation_signature': {'positive_conditions': ['A pending write has multiple requested operations, conditional consent or a stated cost constraint, or an information detour interrupted its approval.'],
                                 'negative_conditions': ['There is no pending state-changing operation.'],
                                 'interaction_phase': ['option_quote', 'authorization', 'before_write']},
    }
    mutation = {
        'analysis': 'Human-authored scope/constraint repair; not an automatically evolved or promoted skill.',
        'semantic_family': 'transaction_consent_scope', 'target_cluster_id': 'human-observed-scope-ledger-v1',
        'operation': 'add', 'target_skill_id': None, 'skill': service,
        'evidence_task_ids': ['7', '12'], 'protected_success_task_ids': ['15'],
        'root_cause_hypothesis': 'Service can execute a second upgrade outside the latest specific consent, or lose a previously stated package cap. Customer fidelity in the archived example is disputed; it motivates a diagnostic, not discovery credit.',
        'evidence_refs': refs, 'expected_effect': 'Fewer extra writes and cap-conflicting writes without repeated confirmations or changes to native scores.',
        'regression_risk': 'Extra clarification or conservative interpretation can hurt completion; inspect activation and unchanged successful operations.',
        'substantive_delta_from_prior': 'Reconcile per-operation scope and pending constraints before write instead of a generic policy reminder.',
    }
    validate_mutation(mutation)
    memory = apply_v2_mutation(ServiceSkillMemoryV2(), mutation, next_skill_id_number=1)
    base = yaml.safe_load((PROJECT / 'configs/airline-global-e-v1-hybrid-seeds-p2.yaml').read_text())
    validate_skill_budgets(memory, base['experiment']['skill_evolution_v2']['skill_budgets'])
    for name, value in [('customer-skill.json', customer), ('service-mutation.json', mutation),
                        ('service-memory.json', memory.to_dict()), ('evidence-rows.json', rows)]:
        (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    (ROOT / 'customer-overlay.txt').write_text(skill.compile().text + '\n')
    (ROOT / 'provenance.json').write_text(json.dumps({
        'author': 'Codex assistant / manually authored', 'protocol': 'human_skill_factorial_v1',
        'not_auto_evolution': True, 'not_promoted': True, 'author_has_seen_E_native_evaluation_definitions': True,
        'fair_evolver_comparison': False, 'no_V_H_content_used': True,
        'independent_semantic_review': 'not_performed',
        'scope': 'Engineering diagnostic; no formal selection, V gate or repair pair credit.',
    }, indent=2) + '\n')
    raw = deepcopy(base)
    identifier = 'evotau-airline-human-scope-ledger-v1-e3-factorial-p2-20261011'
    raw['experiment'].update(id=identifier, generations=1, customer_candidates=1,
                            run_validation=False, run_heldout=False,
                            output_path='experiments/runs/' + identifier,
                            checkpoint_path='experiments/checkpoints/' + identifier + '.json')
    raw['experiment']['task_selection']['evolution'] = ['7', '12', '15']
    raw['launch_readiness'] = {'trial_type': 'human_authored_mechanism_diagnostic', 'formal_promotion': False}
    # Freeze stimuli, including author provenance; not generated reports or this reproducibility script.
    names = ['customer-skill.json', 'service-mutation.json', 'service-memory.json',
             'evidence-rows.json', 'customer-overlay.txt', 'provenance.json']
    raw['manual_skill_trial'] = {
        'protocol_version': 'human_skill_factorial_v1',
        'inputs': {str((ROOT / name).relative_to(PROJECT)): hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names},
        'conditions': ['C0S0', 'C1S0', 'C0S1', 'C1S1'], 'episodes': 12, 'seed': 1,
        'no_score_imports': True, 'automatic_promotion': False,
    }
    launcher = Path('experiments/execution/run-human-skill-factorial.py')
    raw['manual_skill_trial']['inputs'][str(launcher)] = hashlib.sha256((PROJECT / launcher).read_bytes()).hexdigest()
    (PROJECT / 'configs/airline-human-scope-ledger-v1-e3-factorial-p2.yaml').write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
    print(json.dumps({'customer_id': skill.candidate_id, 'schema_valid': True, 'service_skills': len(memory.skills), 'episodes': 12}))


if __name__ == '__main__':
    main()
