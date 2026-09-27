import pytest


@pytest.fixture(autouse=True)
def run_in_tmp_path(tmp_path, monkeypatch):
    """Run each test in its own directory.

    MotionCorrect.motion_correct(save_movie=True) writes to the working directory, naming the
    file by the time to the second, so tests running in parallel (pytest -n) could otherwise
    write to the same file.
    """
    monkeypatch.chdir(tmp_path)
