from ravel_hls.domain.graph import NumericType
from ravel_hls.planning.chain import Candidate, Cost, StreamContract, resolve_chain


def test_chain_planning_retains_a_slower_cheaper_stage_when_the_next_stage_is_the_bottleneck():
    numeric = NumericType("fixed", 8, 4, True, "RND", "SAT_SYM")
    stream = StreamContract("features", (4, 2), numeric, 2)
    expensive = Candidate("fast", 1, ("conv",), stream, stream, Cost(10, 90, 10), True)
    cheap = Candidate("small", 1, ("conv",), stream, stream, Cost(12, 10, 12), True)
    following = Candidate("following", 1, ("head",), stream, stream, Cost(20, 20, 20), False)

    plan = resolve_chain(((expensive, cheap), (following,)))

    assert plan.findings == ()
    assert tuple(stage.id for stage in plan.stages) == ("small", "following")
    assert plan.cost == Cost(20, 30, 32)
    reversed_plan = resolve_chain(((cheap, expensive), (following,)))
    assert reversed_plan == plan


def test_stream_bridge_preserves_integer_codes_and_masks_only_padding():
    from ravel_hls.planning.bridges import bridge_for

    numeric = NumericType("fixed", 8, 4, True, "RND", "SAT_SYM")
    source = StreamContract("features", (5,), numeric, 2)
    target = StreamContract("features", (5,), numeric, 3)
    bridge = bridge_for(source, target)

    assert bridge is not None
    assert bridge.transfer(((1, -2), (3, 4), (5, 99))) == ((1, -2, 3), (4, 5, 0))
    assert bridge.events == ((0, "consume"), (2, "consume"), (2, "produce"), (4, "consume"), (4, "produce"))


def test_planner_inserts_a_qualified_bridge_and_rejects_truncating_candidate_bounds():
    numeric = NumericType("fixed", 8, 4, True, "RND", "SAT_SYM")
    packed = StreamContract("features", (6,), numeric, 3)
    narrow = StreamContract("features", (6,), numeric, 2)
    producer = Candidate("producer", 1, ("conv",), packed, packed, Cost(4, 2, 4), True)
    consumer = Candidate("consumer", 1, ("dense",), narrow, narrow, Cost(5, 3, 5), True)

    plan = resolve_chain(((producer,), (consumer,)))

    assert not plan.findings
    assert len(plan.bridges) == 1
    bounded = resolve_chain(((producer, producer), (consumer,)), max_candidates=1)
    assert bounded.findings[0].code == "planner.candidate_bound"
    assert bounded.stages == ()


def test_bridge_schedule_moves_shared_lanes_together_and_drains_a_partial_word():
    from ravel_hls.planning.bridges import bridge_for

    numeric = NumericType("fixed", 8, 4, True, "RND", "SAT_SYM")
    bridge = bridge_for(StreamContract("features", (5,), numeric, 4),
                        StreamContract("features", (5,), numeric, 2))
    assert bridge.to_dict()["schedule"]["cycles"] == 3
    assert bridge.events == ((0, "consume"), (0, "produce"), (1, "produce"), (2, "consume"), (2, "produce"))
    assert bridge.transfer(((11, 12, 13, 14), (15, 99, 99, 99))) == ((11, 12), (13, 14), (15, 0))


def test_planner_ranks_unknown_cycle_estimates_conservatively_without_making_them_illegal():
    numeric = NumericType("fixed", 8, 4, True, "RND", "SAT_SYM")
    stream = StreamContract("features", (8,), numeric, 2)
    uncalibrated = Candidate("uncalibrated", 1, ("stage",), stream, stream, Cost(4, 1, 4), True, "uncalibrated-native")
    calibrated = Candidate("calibrated", 1, ("stage",), stream, stream, Cost(8, 2, 8), True, "calibrated")
    assert resolve_chain(((uncalibrated, calibrated),)).stages == (calibrated,)
    assert resolve_chain(((uncalibrated,),)).findings == ()


def test_chain_uses_only_the_registered_bridge_capabilities():
    numeric = NumericType("fixed", 8, 4, True, "RND", "SAT_SYM")
    packed = StreamContract("features", (6,), numeric, 3)
    narrow = StreamContract("features", (6,), numeric, 2)
    producer = Candidate("producer", 1, ("conv",), packed, packed, Cost(4, 2, 4), True)
    consumer = Candidate("consumer", 1, ("dense",), narrow, narrow, Cost(5, 3, 5), True)
    disabled = resolve_chain(((producer,), (consumer,)), bridge_strategies=())
    assert disabled.stages == ()
    assert disabled.findings[0].code == "planner.no_qualified_plan"
