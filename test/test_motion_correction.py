import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile
import h5py

import jnormcorre.motion_correction
from jnormcorre.simulation import SimData
from jnormcorre.motion_correction import MotionCorrect
from jnormcorre.utils import registrationarrays


class Test_Simulation:

    @pytest.mark.parametrize("frames", [10, 50, 100])
    @pytest.mark.parametrize("X", [30, 60, 90, 120])
    @pytest.mark.parametrize("Y", [30, 60, 90, 120])
    def test_init(self, frames, X, Y):
        sim = SimData(frames=frames, X=X, Y=Y, n_blobs=10, noise_amplitude=0.2,
                      blob_amplitude=5, max_drift=(0.0001, 0.01), max_jitter=1,
                      background_noise=1, shot_noise=0.2)

    @pytest.mark.parametrize("frames", [10, 50, 100])
    @pytest.mark.parametrize("X", [30, 60, 90, 120])
    @pytest.mark.parametrize("Y", [30, 60, 90, 120])
    def test_run(self, frames, X, Y):
        sim = SimData(frames=frames, X=X, Y=Y, n_blobs=10, noise_amplitude=0.2,
                      blob_amplitude=5, max_drift=(0.0001, 0.01), max_jitter=1,
                      background_noise=1, shot_noise=0.2)

        _, _ = sim.simulate()


class Test_mc:

    def save_sample(self, input_, data, h5_loc=None):

        if isinstance(input_, np.ndarray):
            return input_

        input_ = Path(input_)

        # save temp file
        if input_.suffix in ('.tiff', '.tif'):
            tifffile.imwrite(input_, data=data)

        elif input_.suffix in ('.h5', '.hdf5'):

            with h5py.File(input_.as_posix(), "w") as h:
                # h.create_group("data")
                ds = h.create_dataset(h5_loc, data=data, shape=data.shape, dtype=data.dtype)
                ds[:] = data[:]

        else:
            raise ValueError(f"unknown file extension: {input_.suffix}")

        return input_

    def setup_method(self):

        self.frames, self.X, self.Y = (25, 95, 85)
        sim = SimData(frames=self.frames, X=self.X, Y=self.Y, n_blobs=10, noise_amplitude=0.2,
                      blob_amplitude=5, max_drift=(0.0001, 0.01), max_jitter=1,
                      background_noise=1, shot_noise=0.2)

        data, shifts = sim.simulate()
        self.data = data
        self.shifts = shifts

    @pytest.mark.parametrize("file_type", [("test.tiff", ""), ("test.h5", "data"),
                                           ("test.h5", "data/ch0"), ("test.h5", "data/ch0/dff")])
    def test_file(self, file_type):

        name, h5_loc = file_type

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            # create sim data
            data, shifts = self.data, self.shifts
            input_ = self.save_sample(tmp_dir.joinpath(name), data, h5_loc=h5_loc)

            if name.split(".")[-1] == "tiff":
                lazy_dataset = registrationarrays.TiffArray(input_)
            else:
                lazy_dataset = registrationarrays.Hdf5Array(input_, h5_loc)

            mc = MotionCorrect(lazy_dataset,
                               max_shifts=(6, 6), niter_rig=4, max_deviation_rigid=3,
                               upsample_factor_grid=4, strides=(50, 50), frames_per_split=1000,
                               num_splits_to_process_els=5, num_splits_to_process_rig=5, pw_rigid=True,
                               overlaps=(10, 10), min_mov=-5, niter_els=1)

            # Perform motion correction
            mc.motion_correct(save_movie=True)
            self.shifts = mc.shifts_rig

    @pytest.mark.parametrize("max_shifts", [(6, 6), (39, 39)])
    @pytest.mark.parametrize("num_splits_to_process_rig", [5])
    @pytest.mark.parametrize("num_splits_to_process_els", [5])
    @pytest.mark.parametrize("frames_per_split", [100, 1000])
    @pytest.mark.parametrize("overlaps", [(10, 10), (24, 24)])
    @pytest.mark.parametrize("pw_rigid", [True, False])
    @pytest.mark.parametrize("min_mov", [None, -5, 5])
    @pytest.mark.parametrize("niter_els", [1, 3])
    def test_parameters(self, max_shifts, num_splits_to_process_rig, num_splits_to_process_els,
                        overlaps, pw_rigid, frames_per_split, min_mov, niter_els,
                        niter_rig=4, max_deviation_rigid=3, upsample_factor_grid=4,
                        strides=(50, 50)):

        input_, shifts = self.data, self.shifts

        # Create MotionCorrect instance
        mc = MotionCorrect(input_,
                           max_shifts=max_shifts, niter_rig=niter_rig, frames_per_split=frames_per_split,
                           num_splits_to_process_rig=num_splits_to_process_rig, strides=strides, overlaps=overlaps,
                           pw_rigid=pw_rigid,
                           num_splits_to_process_els=num_splits_to_process_els,
                           upsample_factor_grid=upsample_factor_grid, max_deviation_rigid=max_deviation_rigid,
                           min_mov=min_mov, niter_els=niter_els)

        # Perform motion correction
        mc.motion_correct(save_movie=True)
        calc_shifts = mc.shifts_rig

        # test correct shift reconstruction
        np.allclose(shifts, calc_shifts), f"calculated shifts are to different from True value"

    @pytest.mark.parametrize("num_splits_to_process_els", [5, 10])
    @pytest.mark.parametrize("splits_els", [5, 10])
    @pytest.mark.parametrize("niter_els", [1, 3])
    def test_els_split(self, num_splits_to_process_els, splits_els, niter_els, pw_rigid=True):

        input_, shifts = self.data, self.shifts

        # Create MotionCorrect instance
        mc = MotionCorrect(input_,
                           max_shifts=(6, 6), niter_rig=4, frames_per_split=500,
                           num_splits_to_process_rig=5, strides=(50, 50), overlaps=(10, 10),
                           pw_rigid=pw_rigid,
                           num_splits_to_process_els=num_splits_to_process_els,
                           upsample_factor_grid=4, max_deviation_rigid=3,
                           min_mov=-1, niter_els=niter_els)

        # Perform motion correction
        mc.motion_correct(save_movie=True)

    @pytest.mark.parametrize("num_splits_to_process_rig", [5, 10])
    @pytest.mark.parametrize("frames_per_split", [500, 1000])
    @pytest.mark.parametrize("niter_els", [1, 3])
    def test_rig_split(self, num_splits_to_process_rig, frames_per_split, niter_els, pw_rigid=True):

        input_, shifts = self.data, self.shifts

        # Create MotionCorrect instance
        mc = MotionCorrect(input_,
                           max_shifts=(6, 6), niter_rig=4, frames_per_split=frames_per_split,
                           num_splits_to_process_rig=num_splits_to_process_rig, strides=(50, 50), overlaps=(10, 10),
                           pw_rigid=pw_rigid, num_splits_to_process_els=5,
                           upsample_factor_grid=4, max_deviation_rigid=3,
                           min_mov=-1, niter_els=niter_els)

        # Perform motion correction
        mc.motion_correct(save_movie=True)

    @pytest.mark.parametrize("n_frames", [25, 100])
    @pytest.mark.parametrize("pw_rigid", [True, False])
    @pytest.mark.parametrize("max_drift", [(1e-1), (1e-4, 1e-2), (1e-3, 1e-2), (1e-3, 1e-1)])
    def test_movement(self, n_frames, pw_rigid, max_drift):

        frames, X, Y = (n_frames, self.X, self.Y)
        max_shift = int(min(X, Y) / 2) - 1

        # simulate movement artifacts
        sim = SimData(frames=frames, X=X, Y=Y, n_blobs=10, noise_amplitude=0.2,
                      blob_amplitude=5, max_drift=max_drift, max_jitter=1,
                      background_noise=1, shot_noise=0.2)

        data, shifts = sim.simulate()
        assert np.max(shifts) < max_shift, f"simulated data deviates too much"

        # Create MotionCorrect instance

        mc = MotionCorrect(data,
                           max_shifts=(max_shift, max_shift), niter_rig=4, frames_per_split=1000,
                           num_splits_to_process_rig=5, strides=(50, 50), overlaps=(10, 10),
                           pw_rigid=pw_rigid, num_splits_to_process_els=5,
                           upsample_factor_grid=4, max_deviation_rigid=3,
                           min_mov=-1, niter_els=3)

        # Perform motion correction
        mc.motion_correct(save_movie=True)
        calc_shifts = mc.shifts_rig

        # test correct shift reconstruction
        np.allclose(shifts, calc_shifts), f"calculated shifts are to different from True value"

    @pytest.mark.xfail(reason="movement that exceeds capabilities")
    @pytest.mark.parametrize("n_frames", [400])
    @pytest.mark.parametrize("pw_rigid", [True, False])
    @pytest.mark.parametrize("max_drift", [(1e-3, 1e-1), (1e-2, 1e-1)])
    def test_movement_extreme(self, n_frames, pw_rigid, max_drift):

        frames, X, Y = (n_frames, self.X, self.Y)
        max_shift = int(min(X, Y) / 2) - 1

        # simulate movement artifacts
        sim = SimData(frames=frames, X=X, Y=Y, n_blobs=10, noise_amplitude=0.2,
                      blob_amplitude=5, max_drift=max_drift, max_jitter=1,
                      background_noise=1, shot_noise=0.2)

        data, shifts = sim.simulate()
        assert np.max(shifts) < max_shift, f"simulated data deviates too much"

        # Create MotionCorrect instance

        mc = MotionCorrect(data,
                           max_shifts=(max_shift, max_shift), niter_rig=4, frames_per_split=1000,
                           num_splits_to_process_rig=5, strides=(50, 50), overlaps=(10, 10),
                           pw_rigid=pw_rigid, num_splits_to_process_els=5,
                           upsample_factor_grid=4, max_deviation_rigid=3,
                           min_mov=-1, niter_els=3)

        # Perform motion correction
        mc.motion_correct(save_movie=True)
        calc_shifts = mc.shifts_rig

        # test correct shift reconstruction
        np.allclose(shifts, calc_shifts), f"calculated shifts are to different from True value"

    @pytest.mark.parametrize("n_frames", [25, 100])
    @pytest.mark.parametrize("pw_rigid", [True, False])
    @pytest.mark.parametrize("max_drift", [(1e-1), (1e-4, 1e-2), (1e-3, 1e-2), (1e-3, 1e-1)])
    def test_frame_corrector(self, n_frames, pw_rigid, max_drift):

        frames, X, Y = (n_frames, self.X, self.Y)
        max_shift = int(min(X, Y) / 2) - 1

        sim = SimData(frames=frames, X=X, Y=Y, n_blobs=10, noise_amplitude=0.2,
                      blob_amplitude=5, max_drift=max_drift, max_jitter=1,
                      background_noise=1, shot_noise=0.2)

        data, shifts = sim.simulate()
        assert np.max(shifts) < max_shift, f"simulated data deviates too much"

        mc = MotionCorrect(data,
                           max_shifts=(max_shift, max_shift), niter_rig=4, frames_per_split=1000,
                           num_splits_to_process_rig=5, strides=(50, 50), overlaps=(10, 10),
                           pw_rigid=pw_rigid, num_splits_to_process_els=5,
                           upsample_factor_grid=4, max_deviation_rigid=3,
                           min_mov=-1, niter_els=3)

        # Perform motion correction
        registration_object, target_file = mc.motion_correct(save_movie=True)

        saved_dataset = tifffile.imread(target_file).astype(np.float32)

        if mc.pw_rigid:
            template = mc.total_template_els
        else:
            template = mc.total_template_rig
        registration_object = jnormcorre.motion_correction.FrameCorrector(template, mc.max_shifts,
                                                                          mc.strides, mc.overlaps,
                                                                          mc.max_deviation_rigid, min_mov=mc.min_mov)

        registered_data = registration_object.register_frames(saved_dataset)
        #Verify that the registration object gives you the same results as the
        np.allclose(saved_dataset, registered_data), f"calculated shifts are to different from True value"

    @pytest.mark.parametrize("n_frames", [25, 100])
    @pytest.mark.parametrize("pw_rigid", [True, False])
    @pytest.mark.parametrize("max_drift", [(1e-1), (1e-4, 1e-2), (1e-3, 1e-2), (1e-3, 1e-1)])
    def test_registration_object(self, n_frames, pw_rigid, max_drift):

        frames, X, Y = (n_frames, self.X, self.Y)
        max_shift = int(min(X, Y) / 2) - 1

        sim = SimData(frames=frames, X=X, Y=Y, n_blobs=10, noise_amplitude=0.2,
                      blob_amplitude=5, max_drift=max_drift, max_jitter=1,
                      background_noise=1, shot_noise=0.2)

        data, shifts = sim.simulate()
        assert np.max(shifts) < max_shift, f"simulated data deviates too much"

        mc = MotionCorrect(data,
                           max_shifts=(max_shift, max_shift), niter_rig=4, frames_per_split=1000,
                           num_splits_to_process_rig=5, strides=(50, 50), overlaps=(10, 10),
                           pw_rigid=pw_rigid, num_splits_to_process_els=5,
                           upsample_factor_grid=4, max_deviation_rigid=3,
                           min_mov=-1, niter_els=3)

        # Perform motion correction
        registration_obj, target_file = mc.motion_correct(save_movie=True)

        saved_dataset = tifffile.imread(target_file).astype(np.float32)

        registration_arr = jnormcorre.utils.registrationarrays.RegistrationArray(registration_obj, data)

        num_frames = min(registration_arr.shape[0], 50)
        registered_data = registration_arr[:num_frames, :, :]
        #Verify that the registration object gives you the same results as the
        np.allclose(saved_dataset[:num_frames, :, :], registered_data), f"calculated shifts are to different from True value"


class Test_known_shifts:
    """Motion correction recovers the shifts of a movie made by shifting one image."""

    def make_movie(self):
        from scipy.ndimage import gaussian_filter

        rng = np.random.default_rng(0)
        image = gaussian_filter(rng.random((96, 80)), 2) * 100
        shifts = rng.integers(-5, 6, size=(60, 2))
        movie = np.stack([np.roll(image, tuple(s), axis=(0, 1)) for s in shifts])
        movie += rng.normal(0, 0.1, movie.shape)
        return movie.astype(np.float32), shifts

    @pytest.mark.parametrize("pw_rigid", [False, True])
    def test_recovers_rigid_shifts(self, pw_rigid):
        movie, shifts = self.make_movie()
        mc = MotionCorrect(movie, max_shifts=(8, 8), niter_rig=2, pw_rigid=pw_rigid,
                           strides=(32, 32), overlaps=(16, 16), niter_els=1)
        corrector, _ = mc.motion_correct()

        # The estimated shifts undo the movie's shifts, up to the template's own offset.
        error = np.asarray(mc.shifts_rig) + shifts
        error -= np.median(error, axis=0)
        assert np.abs(error).max() < 0.5

        # Corrected frames match the first; before correction, most don't.
        def correlations_with_first_frame(m):
            return np.corrcoef(m[:, 10:-10, 10:-10].reshape(len(m), -1))[0]

        assert np.median(correlations_with_first_frame(movie)) < 0.5
        corrected = np.asarray(corrector.register_frames(movie, pw_rigid=pw_rigid))
        assert correlations_with_first_frame(corrected).min() > 0.95


@pytest.mark.parametrize("pw_rigid", [False, True])
def test_frames_per_split(pw_rigid):
    """Both the rigid and piecewise-rigid passes register the frames in splits of this size."""
    movie = np.random.default_rng(0).random((250, 64, 64)).astype(np.float32)
    mc = MotionCorrect(movie, max_shifts=(5, 5), frames_per_split=100, pw_rigid=pw_rigid,
                       strides=(24, 24), overlaps=(8, 8))
    mc.motion_correct()
    assert len(mc.templates_rig) == 3  # one template per split: frames 0-99, 100-199, 150-249
    if pw_rigid:
        assert len(mc.templates_els) == 3


@pytest.mark.parametrize("pw_rigid", [False, True])
def test_batching(monkeypatch, pw_rigid):
    """batching sets how many frames are registered at a time, without changing the results."""
    rng = np.random.default_rng(0)
    movie = np.stack([np.roll(rng.random((64, 64)), shift, axis=0) for shift in range(-3, 3)] * 5)
    movie = movie.astype(np.float32)

    def run(batching):
        name = "register_frames_to_template_pwrigid" if pw_rigid else "register_frames_to_template_rigid"
        original = getattr(jnormcorre.motion_correction, name)
        sizes = []

        def recording(imgs, *args):
            sizes.append(imgs.shape[0])
            return original(imgs, *args)

        with monkeypatch.context() as m:
            m.setattr(jnormcorre.motion_correction, name, recording)
            mc = MotionCorrect(movie, max_shifts=(5, 5), frames_per_split=20, pw_rigid=pw_rigid,
                               strides=(24, 24), overlaps=(8, 8), batching=batching)
            mc.motion_correct()
        template = mc.total_template_els if pw_rigid else mc.total_template_rig
        return sizes, np.asarray(template)

    default_sizes, default_template = run(None)
    sizes, template = run(7)
    assert set(default_sizes) == {20}  # the default here: a whole split (frames 0-19, 10-29)
    assert max(sizes) == 7 and sum(sizes) == sum(default_sizes)
    np.testing.assert_allclose(template, default_template, rtol=1e-5, atol=1e-5)
