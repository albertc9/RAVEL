import numpy as np

from ravel_hls.domain.graph import NumericType
from ravel_hls.verification.rtl_vectors import write_rtl_vectors


def test_rtl_vectors_encode_canonical_bits_in_little_lane_order_and_mask_padding(tmp_path):
    # Q4.4: -0.5 is 0xf8, +0.75 is 0x0c. Two 8-bit scalar slots form 0x0cf8.
    record = write_rtl_vectors(tmp_path, np.array([[-0.5, 0.75, 0.0]]), np.array([[-0.25]]),
                              NumericType("fixed", 8, 4, True, "RND", "SAT_SYM"),
                              NumericType("fixed", 9, 4, True, "RND", "SAT_SYM"), 2)
    assert (tmp_path / "rtl_input_words.hex").read_text().splitlines() == ["0cf8", "0000"]
    assert (tmp_path / "rtl_expected_words.hex").read_text().splitlines() == ["01f8"]
    assert record["input_words_per_inference"] == 2
    assert record["reference"] == "clean-hls4ml"
