from __future__ import annotations

from runtime.codec import codec_frame_bounds, codec_frame_mask


def test_codec_timeline_alternates_without_drift() -> None:
    counts = []
    for unit_index in range(20):
        start, end = codec_frame_bounds(unit_index * 500, 500, 75)
        counts.append(end - start)
    assert counts == [37, 38] * 10
    assert sum(counts) == 750


def test_codec_mask_uses_absolute_timeline() -> None:
    assert codec_frame_mask(0, 500, 75, 38).sum().item() == 37
    assert codec_frame_mask(500, 500, 75, 38).sum().item() == 38
