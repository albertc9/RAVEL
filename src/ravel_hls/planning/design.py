"""Resolve immutable frontend facts through capability and plan services."""
from ..analysis.phara import PHARA_HYBRID_DSP_PRODUCT_BUDGET, analyze_direct_parameters, analyze_hybrid_parameters
from ..domain.graph import GraphFacts
from ..domain.temporal import recognize_temporal_chain
from ..profiles.aria.plan import build_implementation_plan
from ..planning.temporal import plan_temporal_chain
from ..compatibility.legacy_design import _parameter_bindings, _predicted_interface, _rendering_contract
from ..manifest import canonical_sha256

def resolve_model_design(generation, facts: GraphFacts, frontend_provenance, choices, parameter_payload, native, dense_facts):
    model_facts = facts.to_dict()
    model_family, applicability = generation.match_model_family(
        model_facts, frontend_provenance
    )

    resolved_design = None
    recognition = recognize_temporal_chain(facts)
    if model_family is not None and len(recognition.chain.blocks) <= 2:
        plan = build_implementation_plan(choices, {**model_facts, **dense_facts})
        strategy = generation.strategy(
            "phara" if "phara" in plan else "aria-wide-stream",
            1 if "phara" in plan else 2,
        )
        strategy_findings = strategy.evaluate(
            model_facts["operations"], choices, plan
        )
        if strategy_findings:
            applicability = {
                "status": "unsupported",
                "findings": strategy_findings,
            }
        else:
            interface = _predicted_interface(model_facts, plan)
            coefficient_realization = None
            if "phara" in plan:
                coefficient_realization = (
                    analyze_hybrid_parameters(
                        model_facts,
                        parameter_payload,
                        dsp_product_budget=PHARA_HYBRID_DSP_PRODUCT_BUDGET,
                    )
                    if plan["phara"]["realization"] == "hybrid"
                    else analyze_direct_parameters(model_facts, parameter_payload)
                )
            resolved_design = generation.resolver.resolve(
                model_facts=model_facts,
                implementation_plan=plan,
                interfaces=interface,
                parameter_bindings=_parameter_bindings(
                    model_facts, parameter_payload
                ),
                rendering=_rendering_contract(native, plan),
                coefficient_realization=coefficient_realization,
            )
    multi_report = {}
    if model_family is not None and model_family["id"] == "hgq-temporal-block-chain":
        multi_report["recognition"] = recognition.chain.to_dict()
        outside_release = len(recognition.chain.blocks) > 2
        if outside_release:
            applicability = {"status": "unsupported", "findings": [{
            "code": "family.support.block_count",
            "severity": "error", "operation_id": None,
            "message": "Aria 1.7 qualifies only one- and two-block plans",
        }]}
        elif resolved_design is not None:
            composed = plan_temporal_chain(
                recognition.chain, temporal_packing=plan["temporal_pack"],
                dense_parallelism=plan["dense_parallelism"], input_strategy=strategy.id,
                input_cycles=plan.get("phara", {}).get("stage_cycles", {}).get("fused_region", plan["input_words_per_inference"]),
                dense_cycles=plan["dense_steps"], strategies=generation.stage_strategies,
            )
            if composed.findings:
                resolved_design = None
                applicability = {"status": "unsupported", "findings": [item.to_dict() for item in composed.findings]}
            else:
                resolved_design.update(
                    model_family=model_family, strategy={"id": "aria-composed", "version": 1},
                    resolver={"id": generation.chain_resolver.id, "version": generation.chain_resolver.version},
                    components={"stage_strategies": [entry.to_dict() for entry in generation.stage_strategies],
                                "bridge_strategies": [{"id": entry.id, "version": entry.version} for entry in generation.bridge_strategies],
                                "cost_policy": {"id": "temporal-lexicographic", "version": 1, "unknown_cycles": "rank-after-finite-estimates"}},
                    stages=[item.to_dict() for item in composed.stages],
                    bridges=[item.to_dict() for item in composed.bridges],
                    delegation={"hls4ml_version": "1.2.0", "policy": "native-latency-v1",
                                "settings": {"Strategy": "Latency", "ReuseFactor": 1}},
                    warnings=[{"code": "estimate.uncalibrated_native", "message": "Native stage estimates are analytical lower bounds; vendor measurements are required"}],
                )
                resolved_design["rendering"]["native_operations"] = native
                from ..rendering.vitis.reports import report_bindings
                resolved_design["report_bindings"] = report_bindings(resolved_design)
                resolved_design["rendering"]["dense_filter_lanes"] = recognition.chain.layout.input.shape[-1]
                resolved_design["resolved_design_sha256"] = canonical_sha256({key: value for key, value in resolved_design.items() if key != "resolved_design_sha256"})
    return model_family, applicability, resolved_design, multi_report
