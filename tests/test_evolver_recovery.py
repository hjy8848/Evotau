"""Offline fault injection using the EXACT archived mechanism-dedup visible output."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from test_failure_analysis import (
    AnalystProviders,
    context,
    hypothesis,
    review,
    settings,
)
from test_skill_evolution_v2 import run

from evotau.alternating import LLMAlternatingEvolvers, _parse_evolver_json
from evotau.budget import RequestBudget
from evotau.evolution_candidates import LEGACY_MECHANISM_DEDUP_PROMPT, V2Providers
from evotau.evolver_recovery import (
    DEFAULT_RECOVERY,
    RecoveryExhausted,
    RecoveryIntegrityError,
    UnknownRequestState,
    summarize_recovery,
)
from evotau.failure_analysis import (
    analysis_outcome,
    conservative_dedup_fallback,
    diversity_outcome,
)
from evotau.skill_evolution import _candidate_outcome
from evotau.skill_evolution_config import freeze_v2_policy

INCIDENT = json.loads((Path(__file__).parent / 'fixtures/mechanism-dedup-incident-visible.json').read_text())['visible_text']


def recovery_policy():
    p = settings()
    p['algorithm_version'] = 'analyst_skill_recovery_v4'
    p['evolver_recovery'] = deepcopy(DEFAULT_RECOVERY)
    p['service_evolution']['crossover'] = False
    return p


def provider(tmp_path, monkeypatch, responses, cap=100):
    budget = RequestBudget(cap)
    calls = []

    def dispatch(model, args, prompt, ctx, *, call_name):
        calls.append((call_name, prompt, deepcopy(ctx)))
        def generate():
            value = responses(call_name, ctx, len(calls)) if callable(responses) else responses[len(calls)-1]
            if isinstance(value, BaseException):
                raise value
            return value if isinstance(value, str) else json.dumps(value)
        text = budget.dispatch_external_call(model=model, call_name=call_name, dispatch=generate)
        return _parse_evolver_json(text, call_name=call_name)

    monkeypatch.setattr(LLMAlternatingEvolvers, '_json_call', staticmethod(dispatch))
    v = V2Providers(LLMAlternatingEvolvers(model='offline/mock', model_args={},
                                          request_budget=budget, output_directory=tmp_path))
    v.configure_recovery(recovery_policy(), tmp_path, 'manifest')
    return v, budget, calls


def dedup_context():
    hs = [hypothesis(context(), str(i), f'Correction {i}') for i in range(3)]
    return {'algorithm_version': 'analyst_skill_recovery_v4', 'hypotheses': hs}


def test_valid_original_no_extra_calls(tmp_path, monkeypatch):
    ctx = dedup_context()
    p, b, calls = provider(tmp_path, monkeypatch, [review(ctx['hypotheses'])])
    assert p.deduplicate_failure_hypotheses(ctx) == review(ctx['hypotheses'])
    assert len(calls) == b.snapshot().attempts == 1
    assert p.last_recovery['status'] == 'VALID'


@pytest.mark.parametrize('bad', [INCIDENT, '{"comparisons":[]} {"comparisons":[]}'])
def test_incident_recovers_without_editing_output(tmp_path, monkeypatch, bad):
    ctx = dedup_context()
    p, b, calls = provider(tmp_path, monkeypatch, [bad, review(ctx['hypotheses'])])
    p.deduplicate_failure_hypotheses(ctx)
    assert p.last_recovery['status'] == 'RECOVERED'
    assert len(calls) == b.snapshot().attempts == 2
    attempts = p.last_recovery['attempts']
    assert attempts[0]['request_id'] != attempts[1]['request_id']
    assert attempts[1]['parent_request'] == attempts[0]['request_id']
    assert calls[0][2] == calls[1][2]  # Same frozen scope/data, no extra evidence.
    raw = tmp_path / attempts[0]['provider_call_ref'] / 'visible-completion.json'
    assert json.loads(raw.read_text())['visible_text'] == bad
    assert json.loads(raw.read_text())['semantic_repair'] is False
    # Successful stage/call recovery survives process recreation without extra dispatch.
    new = V2Providers(p.provider)
    new.configure_recovery(recovery_policy(), tmp_path, 'manifest')
    new.deduplicate_failure_hypotheses(ctx)
    assert len(calls) == b.snapshot().attempts == 2
    assert summarize_recovery(tmp_path, 'manifest')['format_recovery_requests'] == 1


def test_exhausted_dedup_allocates_one_not_three(tmp_path, monkeypatch):
    ctx = dedup_context()
    p, b, calls = provider(tmp_path, monkeypatch, [INCIDENT]*3)
    outcome = diversity_outcome(lambda: p.deduplicate_failure_hypotheses(ctx), ctx['hypotheses'])
    assert outcome['status'] == 'DEGRADED'
    assert outcome['recovery']['status'] == 'RECOVERY_EXHAUSTED'
    assert outcome['recovery_attempts'] == 2
    selection = conservative_dedup_fallback(ctx['hypotheses'], outcome)
    assert selection['assigned_count'] == 1
    assert selection['assigned_hypotheses'][0]['mechanism_id'] == '0'
    assert {h['status'] for h in selection['excluded_hypotheses']} == {'DEFERRED_DUE_TO_DEDUP_FAILURE'}
    diversity_outcome(lambda: p.deduplicate_failure_hypotheses(ctx), ctx['hypotheses'])
    assert len(calls) == b.snapshot().attempts == 3


def test_uncertain_not_skill_are_not_forced():
    hs = dedup_context()['hypotheses']
    for i, h in enumerate(hs):
        h['repairability'] = 'uncertain' if i else 'not_skill'
    result = conservative_dedup_fallback(hs, {'reason': 'dedup failed'})
    assert result['assigned_count'] == 0 and result['fallback'] == 'NO_OP'


@pytest.mark.parametrize('invalid', [{'comparisons': []}, {'comparisons': [
    {'left': 'unknown', 'right': '1', 'same_mechanism': False, 'reason': 'x'}]}])
def test_valid_json_invalid_business_contract_exhausts(tmp_path, monkeypatch, invalid):
    ctx = dedup_context()
    p, _, calls = provider(tmp_path, monkeypatch, [INCIDENT, invalid, invalid])
    with pytest.raises(RecoveryExhausted):
        p.deduplicate_failure_hypotheses(ctx)
    assert len(calls) == 3
    assert p.last_recovery['attempts'][1]['error_type'] == 'EvolverSchemaError'


def test_analyst_exhaustion_no_fabricated_cause(tmp_path, monkeypatch):
    p, _, calls = provider(tmp_path, monkeypatch, ['not-json']*3)
    outcome = analysis_outcome(lambda: p.analyze_service_failures(context()), context())
    assert outcome['status'] == 'REJECTED' and outcome['hypotheses'] == []
    assert len(calls) == 3


def test_mutator_exhaustion_rejects_current_only(tmp_path, monkeypatch):
    from test_skill_evolution_v2 import bind_evidence, mutation
    ctx = context()
    p, _, calls = provider(tmp_path, monkeypatch,
                           ['not-json']*3 + [bind_evidence(mutation(operation='no_op'), ctx)])
    result = _candidate_outcome(lambda: p.propose_skill_mutation(ctx), ctx)
    assert result['status'] == 'REJECTED'
    next_ctx = {**ctx, 'proposal_index': 1}
    assert p.propose_skill_mutation(next_ctx)['operation'] == 'no_op'
    assert len(calls) == 4


def test_forged_evidence_not_accepted(tmp_path, monkeypatch):
    ctx = context(); h = hypothesis(ctx)
    h['evidence_refs'][0]['message_sha256'] = 'forged'
    bad = {'hypotheses': [h], 'insufficient_evidence_reason': ''}
    p, _, _ = provider(tmp_path, monkeypatch, [bad]*3)
    with pytest.raises(RecoveryExhausted):
        p.analyze_service_failures(ctx)


@pytest.mark.parametrize('fault', [TimeoutError('unknown submission'), PermissionError('HTTP 401'),
                                   ConnectionError('network failure')])
def test_infrastructure_never_auto_retry(tmp_path, monkeypatch, fault):
    p, _, calls = provider(tmp_path, monkeypatch, [fault])
    with pytest.raises(type(fault)):
        p.deduplicate_failure_hypotheses(dedup_context())
    assert len(calls) == 1
    with pytest.raises(UnknownRequestState):
        p.deduplicate_failure_hypotheses(dedup_context())
    assert len(calls) == 1


@pytest.mark.parametrize('private', [{'user_scenario': 'hidden'}, {'panel': 'H'}, {'validation_episodes': []}])
def test_private_data_fail_closed_before_dispatch(tmp_path, monkeypatch, private):
    p, _, calls = provider(tmp_path, monkeypatch, [])
    with pytest.raises(RecoveryIntegrityError):
        p.analyze_service_failures({**context(), 'leaked': private})
    assert not calls


def test_journal_manifest_and_source_tampering_fail_closed(tmp_path, monkeypatch):
    ctx = dedup_context(); p, _, calls = provider(tmp_path, monkeypatch, [review(ctx['hypotheses'])])
    p.deduplicate_failure_hypotheses(ctx)
    with pytest.raises(ValueError, match='manifest'):
        p.configure_recovery(recovery_policy(), tmp_path, 'different-manifest')
    directory = tmp_path/p.last_recovery['attempts'][0]['provider_call_ref']
    f = directory/'output.json'; f.write_text('{}')
    with pytest.raises(RecoveryIntegrityError):
        p.deduplicate_failure_hypotheses(ctx)
    assert len(calls) == 1


def test_durable_response_before_attempt_publication_reused(tmp_path, monkeypatch):
    ctx = dedup_context(); p, _, calls = provider(tmp_path, monkeypatch, [INCIDENT, review(ctx['hypotheses'])])
    original = p.recovery.journal.freeze
    interrupted = False
    def freeze(name, inputs, callback):
        nonlocal interrupted
        if name.endswith('attempt-1') and not interrupted:
            interrupted = True
            callback()  # Provider response persisted; journal publication interrupted.
            raise RuntimeError('process interruption')
        return original(name, inputs, callback)
    monkeypatch.setattr(p.recovery.journal, 'freeze', freeze)
    with pytest.raises(RuntimeError, match='interruption'):
        p.deduplicate_failure_hypotheses(ctx)
    p.deduplicate_failure_hypotheses(ctx)
    assert len(calls) == 2


def test_pending_request_not_blindly_resent(tmp_path, monkeypatch):
    ctx = dedup_context(); p, _, calls = provider(tmp_path, monkeypatch, [KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        p.deduplicate_failure_hypotheses(ctx)
    directory = p.last_call_directory
    (directory/'failure.json').unlink()  # Simulate death after dispatch, no durable response.
    with pytest.raises(UnknownRequestState):
        p.deduplicate_failure_hypotheses(ctx)
    assert len(calls) == 1


def test_offline_g2_degraded_assignment_resume_no_repeated_episodes(tmp_path, monkeypatch):
    scripted = AnalystProviders(count=3)
    def responses(name, ctx, count):
        if name == 'evotau_customer_evolver': return scripted.customers(ctx, ctx['requested_candidates'])
        if name == 'evotau_customer_semantic_validator': return {**scripted.validate_customer(ctx), 'reason': 'faithful'}
        if name == 'evotau_service_failure_analyst': return scripted.analyze_service_failures(ctx)
        if name == 'evotau_failure_mechanism_dedup': return INCIDENT
        if name == 'evotau_service_skill_mutator': return scripted.propose_skill_mutation(ctx)
        if name == 'evotau_skill_semantic_validator': return {**scripted.validate_skill(ctx), 'reason': 'legal'}
        raise AssertionError(name)
    p, b, calls = provider(tmp_path, monkeypatch, responses)
    with pytest.raises(RuntimeError, match='interruption'):
        run(tmp_path, provider=p, policy=recovery_policy(), validation=True,
            interrupt='service_hypothesis_assignment', generations=2)
    before = len(calls)
    result, _, runner = run(tmp_path, provider=p, policy=recovery_policy(), validation=True, generations=2)
    assert len(result.generations) == 2
    assert calls[before][0] == 'evotau_service_skill_mutator'  # E/Analyst/Dedup reused.
    assert scripted.calls.count('mutate') == 2
    assert len(runner.calls) == len(set(runner.calls))
    count = len(calls), b.snapshot().attempts, len(runner.calls)
    run(tmp_path, provider=p, runner=runner, policy=recovery_policy(), validation=True, generations=2)
    assert count == (len(calls), b.snapshot().attempts, len(runner.calls))
    assert sum(x[0] == 'evotau_failure_mechanism_dedup' for x in calls) == 3  # Identical frozen dedup inputs in Gen1 reuse the exhausted outcome.
    assert all(g['service_after']['skill_count'] == 0 for g in result.generations)


def test_legacy_identity_no_automatic_recovery(tmp_path, monkeypatch):
    p, _, calls = provider(tmp_path, monkeypatch, [INCIDENT])
    p.recovery = None
    from evotau.alternating import EvolverJSONError
    with pytest.raises(EvolverJSONError): p.deduplicate_failure_hypotheses(dedup_context())
    assert len(calls) == 1 and calls[0][1] == LEGACY_MECHANISM_DEDUP_PROMPT


def test_recovery_policy_explicit_version_and_bounds():
    from evotau.evolver_recovery import validate_recovery_policy
    for limit in [-1, 3, True]:
        with pytest.raises(ValueError): validate_recovery_policy({**DEFAULT_RECOVERY, 'max_format_recoveries': limit})
    p = settings(); p['evolver_recovery'] = DEFAULT_RECOVERY
    with pytest.raises(ValueError, match='new'):
        freeze_v2_policy(p, {}, {})


def test_recovery_token_ledger_and_metadata(tmp_path, monkeypatch):
    ctx = dedup_context()
    p, budget, calls = provider(tmp_path, monkeypatch, [])
    texts = [INCIDENT, json.dumps(review(ctx['hypotheses']))]
    def dispatch(model, args, prompt, context, *, call_name):
        text = texts[len(calls)]
        calls.append(call_name)
        payload = {'id': f'mock-{len(calls)}', 'model': model,
                   'choices': [{'finish_reason': 'stop', 'message': {'content': text}}],
                   'usage': {'prompt_tokens': 11, 'completion_tokens': 3}}
        budget.dispatch_external_call(model=model, call_name=call_name, dispatch=lambda: payload)
        return _parse_evolver_json(text, call_name=call_name)
    monkeypatch.setattr(LLMAlternatingEvolvers, '_json_call', staticmethod(dispatch))
    p.deduplicate_failure_hypotheses(ctx)
    summary = summarize_recovery(tmp_path, 'manifest')
    assert budget.snapshot().prompt_tokens == 22
    assert budget.snapshot().completion_tokens == 6
    assert summary['extra_provider_usage']['calls'] == 1
    assert summary['extra_provider_usage']['prompt_tokens'] == 11
    assert summary['extra_provider_usage']['completion_tokens'] == 3
    assert summary['original_valid_rate'] == 0 and summary['recovery_success_rate'] == 1


def test_e10_incident_resume_does_not_rerun_completed_panel(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from test_skill_evolution_v2 import FakeRunner

    from evotau.service_skills import ServiceSkillMemoryV2
    from evotau.skill_evolution import run_skill_evolution_v2
    from evotau.strategies import PromptStrategy

    scripted = AnalystProviders(count=3)
    def responses(name, ctx, count):
        if name == 'evotau_customer_evolver': return scripted.customers(ctx, ctx['requested_candidates'])
        if name == 'evotau_customer_semantic_validator': return {**scripted.validate_customer(ctx), 'reason': 'faithful'}
        if name == 'evotau_service_failure_analyst': return scripted.analyze_service_failures(ctx)
        if name == 'evotau_failure_mechanism_dedup': return INCIDENT
        if name == 'evotau_service_skill_mutator': return scripted.propose_skill_mutation(ctx)
        raise AssertionError(name)
    p, budget, calls = provider(tmp_path, monkeypatch, responses)
    runner = FakeRunner()
    args = {'tasks': {str(i): SimpleNamespace(id=str(i), user_scenario='secret-hidden',
                         description='secret-hidden', user_tools=[]) for i in range(1, 12)},
                'evolution_task_ids': tuple(str(i) for i in range(1, 11)), 'validation_task_ids': ('11',),
                'seed': 1, 'generations': 2, 'customer_candidate_count': 1, 'max_parallel_episodes': 2,
                'evolution_fitness_seed': 1, 'run_validation': True, 'initial_customer': PromptStrategy(''),
                'initial_service': ServiceSkillMemoryV2(), 'runner': runner, 'providers': p,
                'domain_policy': 'Native public policy.', 'output_directory': tmp_path,
                'checkpoint_path': tmp_path/'checkpoint.json', 'manifest_sha256': 'manifest', 'policy': recovery_policy()}
    with pytest.raises(RuntimeError, match='interruption'):
        run_skill_evolution_v2(**args, interrupt_after_stage='service_hypothesis_assignment')
    native_before = tuple(runner.calls)
    first_budget = budget.snapshot().attempts
    result = run_skill_evolution_v2(**args)
    assert len(result.generations) == 2
    assert all(runner.calls.count(cell) == 1 for cell in native_before)
    assert len({cell[0] for cell in native_before}) == 10
    assert sum(c[0] == 'evotau_failure_mechanism_dedup' for c in calls) == 3
    assert budget.snapshot().attempts > first_budget  # Only newly eligible stages consume calls.
    before = len(runner.calls), budget.snapshot().attempts
    run_skill_evolution_v2(**args)
    assert before == (len(runner.calls), budget.snapshot().attempts)
    assignment = json.loads((tmp_path/'evolution-v2/g0000-service_hypothesis_assignment.json').read_text())['payload']
    assert assignment['status'] == 'DEGRADED' and assignment['assigned_count'] == 1
    assert len(result.service.skills) == 0  # No forced promotion.


def test_new_config_only_changes_recovery_identity():
    import yaml

    from evotau.alternating_manifest import AlternatingManifest

    root = Path(__file__).resolve().parents[1]
    old = yaml.safe_load((root/'configs/airline-failure-analyst-hybrid-seeds-p2.yaml').read_text())
    new = yaml.safe_load((root/'configs/airline-analyst-recovery-v4-hybrid-seeds-p2.yaml').read_text())
    a, b = AlternatingManifest.from_mapping(old), AlternatingManifest.from_mapping(new)
    assert a.output_path != b.output_path and a.checkpoint_path != b.checkpoint_path
    assert a.evolution_task_ids == b.evolution_task_ids
    assert a.validation_task_ids == b.validation_task_ids and a.heldout_task_ids == b.heldout_task_ids
    pa, pb = json.loads(a.skill_evolution_v2_json), json.loads(b.skill_evolution_v2_json)
    assert pb.pop('evolver_recovery') == DEFAULT_RECOVERY
    pb['algorithm_version'] = pa['algorithm_version']
    assert pa == pb  # Screen, gate, bootstrap, Bonferroni, skill constraints unchanged.
    assert a.role_models == b.role_models and a.role_model_args == b.role_model_args


def test_recovered_legal_mutation_still_runs_screen_full_e_v_gate(tmp_path, monkeypatch):
    from test_failure_analysis import RepairProviders

    scripted = RepairProviders(count=1)
    mutation_requests = 0
    def responses(name, ctx, count):
        nonlocal mutation_requests
        if name == 'evotau_customer_evolver': return scripted.customers(ctx, ctx['requested_candidates'])
        if name == 'evotau_customer_semantic_validator': return {**scripted.validate_customer(ctx), 'reason': 'faithful'}
        if name == 'evotau_service_failure_analyst': return scripted.analyze_service_failures(ctx)
        if name == 'evotau_service_skill_mutator':
            mutation_requests += 1
            if mutation_requests == 1: return '{bad JSON'
            return scripted.propose_skill_mutation(ctx)
        if name == 'evotau_skill_semantic_validator': return {**scripted.validate_skill(ctx), 'reason': 'legal'}
        raise AssertionError(name)
    p, _, _ = provider(tmp_path, monkeypatch, responses)
    result, _, _ = run(tmp_path, provider=p, policy=recovery_policy(), validation=True, generations=2)
    assert result.generations[0]['service_phase']['candidates'][0]['screen']['passed']
    assert result.generations[0]['service_phase']['candidates'][0]['gate'] is not None
    assert summarize_recovery(tmp_path, 'manifest')['recovered_calls'] == 1


def test_unknown_budget_exhaustion_is_not_format_recovery(tmp_path, monkeypatch):
    from evotau.budget import ProviderBudgetExceeded

    ctx = dedup_context(); p, _, _calls = provider(tmp_path, monkeypatch, [INCIDENT]*3, cap=1)
    with pytest.raises(ProviderBudgetExceeded): p.deduplicate_failure_hypotheses(ctx)
    # No recovery response/candidate was scored, and no fallback was falsely published.
    assert not list((tmp_path/'format-recovery/evolution-v2').glob('*-result.json'))


def test_recovery_journal_digest_and_manual_override_stop(tmp_path, monkeypatch):
    ctx = dedup_context(); p, _, calls = provider(tmp_path, monkeypatch, [review(ctx['hypotheses'])])
    p.deduplicate_failure_hypotheses(ctx)
    final = next((tmp_path/'format-recovery/evolution-v2').glob('*-result.json'))
    original = final.read_text(); doc = json.loads(original)
    doc['payload']['status'] = 'RECOVERED'; final.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='digest'):
        p.deduplicate_failure_hypotheses(ctx)
    final.write_text(original)
    attempt = p.last_recovery['attempts'][0]
    (tmp_path/attempt['provider_call_ref']/'retry-authorization.json').write_text('{}')
    with pytest.raises(RecoveryIntegrityError, match='manual'):
        p.deduplicate_failure_hypotheses(ctx)
    assert len(calls) == 1


def test_no_manifest_switch_changes_freeze_input(tmp_path, monkeypatch):
    ctx = dedup_context(); p, _, calls = provider(tmp_path, monkeypatch, [review(ctx['hypotheses'])])
    p.deduplicate_failure_hypotheses(ctx)
    with pytest.raises(ValueError, match='manifest'):
        p.configure_recovery(recovery_policy(), tmp_path, 'new identity')
    assert p.recovery.manifest_sha == 'manifest'
    p.deduplicate_failure_hypotheses(ctx)
    assert len(calls) == 1
