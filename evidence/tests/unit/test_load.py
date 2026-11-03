import time

from locus.evidence.load import LoadStats


def test_a_wedged_load_keeps_speaking(capsys):
    """A load hung on a GET has nothing arriving to tick a counter."""
    with LoadStats(every_s=0.02).beating():
        time.sleep(0.15)
    said = capsys.readouterr().err
    beats = [line for line in said.splitlines() if line]

    assert len(beats) >= 3, "nothing arrived, and it spoke anyway"
    assert all(b.startswith("load: 0 chunks in") for b in beats)
    assert said == "".join(b + "\n" for b in beats), "every beat is one whole line"
