"""Enumerate typed capabilities and resolve the complete temporal chain."""

from ..domain.temporal import TemporalChain, Finding
from .chain import ChainPlan, Cost, TEMPORAL_RESOLVER
from .calibration import WINDOW_COST_PROFILE
from dataclasses import replace
from .bridges import LOSSLESS_BRIDGES
from .strategies import StageContext, StageRequest, TEMPORAL_STRATEGIES
from .search import SearchReport, SEARCH_CANDIDATE_LIMIT, plan_identity
from .resources import constraint_report, estimate_resources, rejection_reasons


def plan_temporal_chain(chain: TemporalChain, *, temporal_packing: int,
                        dense_parallelism: int, input_strategy: str,
                        input_cycles: int, dense_cycles: int,
                        part=None, clock_period=None, resource_limits=None, target_ii=None,
                        strategies=TEMPORAL_STRATEGIES, bridges=LOSSLESS_BRIDGES, resolver=TEMPORAL_RESOLVER) -> SearchReport:
    context = StageContext(chain.blocks[0].input.id, input_strategy, temporal_packing,
                           dense_parallelism, input_cycles, dense_cycles, part, clock_period)
    findings = []
    def candidates(request):
        evaluations = [strategy.evaluate(request) for strategy in sorted(strategies, key=lambda entry: (entry.id, entry.version))]
        result = tuple(candidate for evaluation in evaluations for candidate in evaluation.candidates)
        findings.extend(finding for evaluation in evaluations for finding in evaluation.findings)
        return result
    domains = []
    for block in chain.blocks:
        domain = candidates(StageRequest(block, context))
        if not domain:
            failed = ChainPlan(findings=tuple(findings) or (Finding("planner.no_stage_candidate", "No qualified strategy can implement this temporal block", block.convolution.id),))
            return SearchReport((), failed, target_ii=target_ii)
        domains.append(domain)
    plans = []
    # Always evaluate the legal incumbent before bounded alternatives. Bounds
    # never authorize selecting an illegal or uncalibrated replacement.
    ordered = sorted(domains[-1], key=lambda candidate: (candidate.implementation is not None, candidate.identity))
    explored = ordered[:SEARCH_CANDIDATE_LIMIT]
    bound_reasons = ["search.candidate_bound"] if len(explored) < len(ordered) else []
    for downstream in explored:
        layouts = candidates(StageRequest(chain.layout, context, previous=downstream.output))
        head = candidates(StageRequest(chain.head, context, layout=chain.layout))
        domain = (*domains[:-1], (downstream,), layouts, head)
        plan = resolver.resolve(domain, bridge_strategies=bridges)
        bound_reasons.extend(finding.code for finding in plan.findings if finding.code.endswith("_bound"))
        if not plan.findings:
            if downstream.confidence == "calibrated":
                cycles = max([stage.cost.cycles + (0 if stage.implementation else 16)
                              for stage in plan.stages]
                             + [WINDOW_COST_PROFILE.bridge_cycles(bridge) for bridge in plan.bridges]) + 1
                plan = replace(plan, cost=Cost(cycles, plan.cost.resource, plan.cost.latency))
            plans.append(plan)
    resources = tuple(estimate_resources(plan, chain) for plan in plans)
    constraints = constraint_report(part, clock_period, resource_limits or {})
    feasible = [plan for plan, estimate in zip(plans, resources) if not rejection_reasons(estimate, constraints)]
    incumbent = next((plan for plan in feasible if not any(stage.implementation for stage in plan.stages)), None)
    calibrated = [plan for plan in feasible if all(stage.confidence in {"analytical", "calibrated"} for stage in plan.stages)]
    def ranking(plan):
        estimate = resources[plans.index(plan)]
        normalized = sum(value / constraints["resource_limits"].get(name, 1)
                         for name, value in estimate.values.items() if value is not None)
        if target_ii is not None and plan.cost.cycles <= target_ii:
            return 0, estimate.lut if estimate.lut is not None else float("inf"), normalized, plan.cost.cycles, plan_identity(plan)
        return 1, plan.cost.cycles, normalized, plan.cost.latency, plan_identity(plan)
    selected = min(calibrated, key=ranking, default=incumbent)
    if selected is None:
        selected = ChainPlan(findings=(*findings, Finding("search.no_feasible_plan", "No selectable plan meets the requested core constraints")))
    bound_reasons.extend(finding.code for finding in findings if finding.code.endswith("_bound"))
    return SearchReport(tuple(plans), selected, complete=not bound_reasons,
                        resources=resources, constraints=constraints,
                        generated=len(ordered), evaluated=len(explored), bound_reasons=tuple(bound_reasons), target_ii=target_ii)
