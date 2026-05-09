import numpy as np

from echoxflow.scan import (
    BeamspacePixelGrid,
    CartesianGrid,
    CartesianPixelGrid,
    ImageLayer,
    SectorDepthRuler,
    SectorGeometry,
    SlicePlane,
    SphericalGeometry,
    VolumeGrid,
    clinical_spherical_mosaic,
    compose_layers,
    opacity_from_values,
    preconverted_spherical_mosaic,
    prepare_3d_brightness_for_display,
    relative_qrs_trigger_times,
    relative_volume_timestamps,
    resize_pixel_xy,
    sector_depth_ticks,
    sector_geometry_from_mapping,
    sector_to_cartesian,
    slice_volume,
)
from echoxflow.scan.beat_stitching import (
    _antisymmetric_cyclic_order,
    _best_antisymmetric_cyclic_rotation,
    _lowest_resolution_spatial_axis,
)
from echoxflow.scan.contours import LV_GROUP_LAYOUT, build_contour_masks


def test_contour_masks_assign_shared_boundary_to_endo_only() -> None:
    endo_vertices = np.asarray([[5, 3], [9, 3], [9, 10], [5, 10]], dtype=np.float32)
    outer_vertices = np.asarray([[3, 1], [11, 1], [11, 12], [3, 12]], dtype=np.float32)
    contour_rows = []
    for endo, outer in zip(endo_vertices, outer_vertices):
        contour_rows.extend((endo + (outer - endo) * (step / 4.0)).tolist() for step in range(5))
    points = np.asarray(contour_rows, dtype=np.float32)

    result = build_contour_masks(points, image_shape=(14, 14), group_layout=LV_GROUP_LAYOUT)

    assert not bool(np.any(result.endo_mask & result.myo_mask))
    assert bool(result.endo_mask[3, 5])
    assert not bool(result.myo_mask[3, 5])


def test_contour_masks_do_not_close_myocardium_across_mitral_valve() -> None:
    endo_vertices = np.asarray([[6, 3], [5, 8], [8, 13], [11, 8], [10, 3]], dtype=np.float32)
    outer_vertices = np.asarray([[3, 1], [2, 8], [8, 15], [14, 8], [13, 1]], dtype=np.float32)
    contour_rows = []
    for endo, outer in zip(endo_vertices, outer_vertices):
        contour_rows.extend((endo + (outer - endo) * (step / 4.0)).tolist() for step in range(5))
    points = np.asarray(contour_rows, dtype=np.float32)

    result = build_contour_masks(points, image_shape=(18, 18), group_layout=LV_GROUP_LAYOUT)

    assert not bool(np.any(result.myo_mask[1:3, 6:11]))
    assert bool(result.myo_mask[8, 3])
    assert bool(result.myo_mask[8, 13])
    assert bool(result.myo_mask[14, 8])
    assert not bool(np.any(result.endo_mask & result.myo_mask))


def test_sector_to_cartesian_preserves_centerline_values() -> None:
    image = np.tile(np.arange(5, dtype=np.float32)[:, None], (1, 5))
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.0,
        depth_end_m=4.0,
        tilt_rad=0.0,
        width_rad=1.0,
        grid_shape=image.shape,
    )
    grid = CartesianGrid(shape=(5, 5), x_range_m=(-1.0, 1.0), y_range_m=(0.0, 4.0))

    converted = sector_to_cartesian(image, geometry, grid=grid, interpolation="nearest")

    assert converted.data.shape == (5, 5)
    assert converted.mask[:, 2].all()
    assert np.allclose(converted.data[:, 2], np.arange(5, dtype=np.float32))


def test_cartesian_grid_from_sector_height_includes_lifted_near_arc() -> None:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.02,
        depth_end_m=0.10,
        tilt_rad=0.0,
        width_rad=1.0,
        grid_shape=(32, 32),
    )

    grid = CartesianGrid.from_sector_height(geometry, 64)

    assert grid.y_range_m[0] < geometry.depth_start_m
    assert np.isclose(grid.y_range_m[0], geometry.depth_start_m * np.cos(0.5))
    assert np.isclose(grid.y_range_m[1], geometry.depth_end_m)


def test_beamspace_pixel_grid_round_trips_physical_points() -> None:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.01,
        depth_end_m=0.09,
        tilt_rad=0.0,
        width_rad=0.8,
        grid_shape=(9, 11),
    )
    grid = BeamspacePixelGrid(geometry=geometry, shape=(9, 11))
    points = np.asarray([[-0.01, 0.04], [0.0, 0.06], [0.02, 0.07]], dtype=np.float32)

    pixels = grid.physical_to_pixel_xy(points)
    restored = grid.pixel_xy_to_physical(pixels)

    np.testing.assert_allclose(restored, points, atol=1e-6)


def test_cartesian_pixel_grid_round_trips_physical_points_and_resizes() -> None:
    cartesian = CartesianPixelGrid(CartesianGrid(shape=(5, 5), x_range_m=(-0.02, 0.02), y_range_m=(0.0, 0.08)))
    points = np.asarray([[-0.02, 0.0], [0.0, 0.04], [0.02, 0.08]], dtype=np.float32)

    pixels = cartesian.physical_to_pixel_xy(points)
    restored = cartesian.pixel_xy_to_physical(pixels)
    resized = resize_pixel_xy(pixels, source_shape=(5, 5), target_shape=(9, 9))

    np.testing.assert_allclose(restored, points, atol=1e-6)
    np.testing.assert_allclose(resized[[0, -1]], [[0.0, 0.0], [8.0, 8.0]], atol=1e-6)


def test_sector_geometry_from_mapping_accepts_center_width_metadata() -> None:
    geometry = sector_geometry_from_mapping(
        {"depth_start_m": 0.01, "depth_end_m": 0.09, "tilt_rad": 0.1, "width_rad": 0.4},
        grid_shape=(8, 10),
    )

    assert geometry.grid_shape == (8, 10)
    assert geometry.depth_start_m == 0.01
    assert geometry.depth_end_m == 0.09
    assert np.isclose(geometry.angle_start_rad, -0.1)
    assert np.isclose(geometry.angle_end_rad, 0.3)


def test_sector_geometry_from_mapping_accepts_alternate_key_casing() -> None:
    geometry = sector_geometry_from_mapping(
        {"DepthStart": 0.01, "DepthEnd": 0.09, "Tilt": 0.1, "Width": 0.4, "GridSize": [8, 10]}
    )

    assert geometry.grid_shape == (8, 10)
    assert geometry.depth_start_m == 0.01
    assert geometry.depth_end_m == 0.09
    assert np.isclose(geometry.angle_start_rad, -0.1)
    assert np.isclose(geometry.angle_end_rad, 0.3)


def test_slice_volume_samples_physical_plane() -> None:
    volume = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    grid = VolumeGrid(origin_m=(0.0, 0.0, 0.0), spacing_m=(1.0, 1.0, 1.0))
    plane = SlicePlane(
        origin_m=(0.0, 0.0, 2.0),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 1.0, 0.0),
        u_range_m=(0.0, 3.0),
        v_range_m=(0.0, 3.0),
        shape=(4, 4),
    )

    sliced = slice_volume(volume, grid, plane, interpolation="nearest")

    assert sliced.data.shape == (4, 4)
    assert sliced.mask.all()
    assert np.allclose(sliced.data, volume[2])


def test_compose_layers_supports_opacity_masking() -> None:
    base = np.zeros((2, 2), dtype=np.float32)
    overlay = np.ones((2, 2), dtype=np.float32)
    alpha = opacity_from_values(np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))

    rgba = compose_layers(
        [
            ImageLayer(base, alpha=1.0),
            ImageLayer(overlay, alpha=1.0, mask=alpha),
        ],
        background="#222222",
    )

    assert rgba.shape == (2, 2, 4)
    assert rgba[0, 1, 0] > rgba[0, 0, 0]


def test_sector_depth_ticks_follow_selected_sector_side() -> None:
    geometry = SectorGeometry.from_center_width(
        depth_start_m=0.01,
        depth_end_m=0.06,
        tilt_rad=0.0,
        width_rad=0.8,
        grid_shape=(8, 8),
    )

    left_ticks = sector_depth_ticks(geometry, SectorDepthRuler(side="left"))
    right_ticks = sector_depth_ticks(geometry, SectorDepthRuler(side="right"))

    assert [int(tick.depth_cm) for tick in left_ticks] == [2, 3, 4, 5]
    assert any(tick.label == "5" for tick in left_ticks)
    assert all(tick.base_xy_m[0] < 0.0 for tick in left_ticks)
    assert all(tick.base_xy_m[0] > 0.0 for tick in right_ticks)
    assert all(tick.end_xy_m[0] < tick.base_xy_m[0] for tick in left_ticks)
    assert all(tick.end_xy_m[0] > tick.base_xy_m[0] for tick in right_ticks)


def test_spherical_mosaics_return_twelve_temporal_slices() -> None:
    volumes = np.arange(2 * 5 * 7 * 9, dtype=np.uint8).reshape(2, 5, 7, 9)
    geometry = SphericalGeometry(
        depth_start_m=0.01,
        depth_end_m=0.12,
        azimuth_width_rad=0.8,
        elevation_width_rad=0.7,
    )

    pre = preconverted_spherical_mosaic(volumes, output_size=(8, 10))
    clinical = clinical_spherical_mosaic(volumes, geometry, output_size=(8, 10), interpolation="nearest")

    assert pre.frames.shape == (2, 24, 40)
    assert clinical.frames.shape == (2, 24, 40)
    assert float(np.max(pre.frames)) > 0.0
    assert float(np.nanmax(clinical.frames)) > 0.0
    assert np.isnan(clinical.frames).any()


def test_preconverted_spherical_mosaic_first_column_has_radial_axis_vertical() -> None:
    volumes = np.arange(1 * 3 * 5 * 5, dtype=np.float32).reshape(1, 3, 5, 5)

    mosaic = preconverted_spherical_mosaic(volumes, output_size=(5, 3))

    first_row_panel = mosaic.frames[0, 0:5, 0:3]
    middle_row_panel = mosaic.frames[0, 5:10, 0:3]
    bottom_row_panel = mosaic.frames[0, 10:15, 0:3]
    assert first_row_panel.tolist() == volumes[0, :, 3, :].T.tolist()
    assert middle_row_panel.tolist() == volumes[0, :, 2, :].T.tolist()
    assert bottom_row_panel.tolist() == volumes[0, :, 1, :].T.tolist()


def test_prepare_3d_brightness_stitches_beats_along_elevation() -> None:
    volumes = np.arange(8, dtype=np.uint8).reshape(8, 1, 1, 1)
    timestamps = np.asarray([10.05, 10.95, 11.05, 11.95, 12.05, 12.95, 13.05, 13.95], dtype=np.float64)
    metadata = {
        "stitch_beat_count": 4,
        "metadata": {
            "time_reference": {"volume_time_origin_s": 10.0},
            "qrs_trigger_times": [0.0, 1.0, 2.0, 3.0, 4.0],
        },
    }

    prepared = prepare_3d_brightness_for_display(volumes, timestamps, metadata)

    assert prepared.was_stitched is True
    assert prepared.stitch_beat_count == 4
    assert prepared.volumes.shape == (2, 4, 1, 1)
    assert np.allclose(prepared.timestamps, [0.05, 0.95])
    assert prepared.volumes[:, :, 0, 0].tolist() == [[6, 4, 2, 0], [7, 5, 3, 1]]
    np.testing.assert_allclose(
        prepared.source_timestamps,
        [
            [3.05, 2.05, 1.05, 0.05],
            [3.95, 2.95, 1.95, 0.95],
        ],
    )


def test_time_reference_basis_uses_origin_s_and_relative_qrs_axis() -> None:
    metadata = {
        "metadata": {
            "time_reference": {
                "basis": "relative_to_volume_origin",
                "origin_s": 1.3,
            },
            "qrs_trigger_times": [-0.1, 0.8, 1.7],
        },
    }

    volume_timestamps = relative_volume_timestamps(np.asarray([1.3, 1.8, 2.3], dtype=np.float64), metadata)
    qrs_timestamps = relative_qrs_trigger_times(metadata)

    assert np.allclose(volume_timestamps, [0.0, 0.5, 1.0])
    assert np.allclose(qrs_timestamps, [-0.1, 0.8, 1.7])


def test_prepare_3d_brightness_aligns_stitched_beats_by_qrs_elapsed_time() -> None:
    volumes = np.arange(6, dtype=np.uint8).reshape(6, 1, 1, 1)
    timestamps = np.asarray([0.05, 0.5, 0.95, 1.05, 1.5, 1.95], dtype=np.float64)
    metadata = {
        "stitch_beat_count": 2,
        "metadata": {
            "qrs_trigger_times": [0.0, 1.0, 2.0],
        },
    }

    prepared = prepare_3d_brightness_for_display(volumes, timestamps, metadata)

    assert prepared.was_stitched is True
    assert prepared.volumes.shape == (3, 2, 1, 1)
    assert np.allclose(prepared.timestamps, [0.05, 0.5, 0.95])
    assert prepared.volumes[:, :, 0, 0].tolist() == [[3, 0], [4, 1], [5, 2]]
    np.testing.assert_allclose(
        prepared.source_timestamps,
        [
            [1.05, 0.05],
            [1.5, 0.5],
            [1.95, 0.95],
        ],
    )


def test_prepare_3d_brightness_does_not_accelerate_long_rr_intervals() -> None:
    volumes = np.arange(7, dtype=np.uint8).reshape(7, 1, 1, 1)
    timestamps = np.asarray([0.0, 0.5, 0.95, 1.0, 1.5, 1.95, 2.3], dtype=np.float64)
    metadata = {
        "stitch_beat_count": 2,
        "metadata": {
            "qrs_trigger_times": [0.0, 1.0, 2.5],
        },
    }

    prepared = prepare_3d_brightness_for_display(volumes, timestamps, metadata)

    assert prepared.was_stitched is True
    assert prepared.volumes.shape == (3, 2, 1, 1)
    assert np.allclose(prepared.timestamps, [0.0, 0.5, 0.95])
    assert prepared.volumes[:, :, 0, 0].tolist() == [[3, 0], [4, 1], [5, 2]]
    np.testing.assert_allclose(
        prepared.source_timestamps,
        [
            [1.0, 0.0],
            [1.5, 0.5],
            [1.95, 0.95],
        ],
    )
    assert 6 not in prepared.volumes[:, :, 0, 0]


def test_prepare_3d_brightness_stitches_along_smallest_spatial_axis() -> None:
    volumes = np.arange(8 * 4 * 2 * 5, dtype=np.uint8).reshape(8, 4, 2, 5)
    timestamps = np.asarray([10.05, 10.95, 11.05, 11.95, 12.05, 12.95, 13.05, 13.95], dtype=np.float64)
    metadata = {
        "stitch_beat_count": 4,
        "metadata": {
            "time_reference": {"volume_time_origin_s": 10.0},
            "qrs_trigger_times": [0.0, 1.0, 2.0, 3.0, 4.0],
        },
    }

    prepared = prepare_3d_brightness_for_display(volumes, timestamps, metadata)

    assert prepared.was_stitched is True
    assert prepared.volumes.shape == (2, 4, 8, 5)


def test_prepare_3d_brightness_grows_only_the_lowest_resolution_spatial_axis() -> None:
    volumes = np.arange(8 * 4 * 6 * 2, dtype=np.uint8).reshape(8, 4, 6, 2)
    timestamps = np.asarray([0.05, 0.95, 1.05, 1.95, 2.05, 2.95, 3.05, 3.95], dtype=np.float64)
    metadata = {"stitch_beat_count": 4, "metadata": {"qrs_trigger_times": [0.0, 1.0, 2.0, 3.0, 4.0]}}

    prepared = prepare_3d_brightness_for_display(volumes, timestamps, metadata)

    assert prepared.was_stitched is True
    assert _lowest_resolution_spatial_axis(volumes) == 2
    assert prepared.volumes.shape == (2, 4, 6, 8)


def test_antisymmetric_cyclic_order_is_the_only_candidate_family() -> None:
    assert _antisymmetric_cyclic_order(0, 3, 0) == (2, 1, 0)
    assert _antisymmetric_cyclic_order(0, 3, 1) == (1, 0, 2)
    assert _antisymmetric_cyclic_order(0, 4, 0) == (3, 2, 1, 0)
    assert _antisymmetric_cyclic_order(0, 6, 0) == (5, 4, 3, 2, 1, 0)
    assert {_antisymmetric_cyclic_order(0, 4, offset) for offset in range(4)} == {
        (3, 2, 1, 0),
        (2, 1, 0, 3),
        (1, 0, 3, 2),
        (0, 3, 2, 1),
    }


def test_best_antisymmetric_cyclic_rotation_tie_breaks_to_zero() -> None:
    volumes = np.zeros((4, 2, 2, 2), dtype=np.float32)
    groups = [(0, [[0], [1], [2], [3]])]

    offset = _best_antisymmetric_cyclic_rotation(volumes, groups, stitch_axis=0, beat_count=4)

    assert offset == 0


def test_prepare_3d_brightness_selects_zero_offset_for_two_synthetic_windows() -> None:
    volumes, timestamps = _synthetic_three_beat_windows(((150.0, 200.0), (100.0, 150.0), (50.0, 100.0)))
    metadata = {"stitch_beat_count": 3, "metadata": {"qrs_trigger_times": np.arange(7, dtype=np.float64)}}

    prepared = prepare_3d_brightness_for_display(volumes, timestamps, metadata)

    assert prepared.was_stitched is True
    assert prepared.volumes.shape == (4, 6, 4, 4)
    np.testing.assert_allclose(prepared.timestamps, [0.05, 0.95, 3.05, 3.95])
    _assert_elevation_planes(prepared.volumes, [50.0, 100.0, 100.0, 150.0, 150.0, 200.0])


def test_prepare_3d_brightness_selects_rotated_offset_for_two_synthetic_windows() -> None:
    volumes, timestamps = _synthetic_three_beat_windows(((100.0, 150.0), (50.0, 100.0), (150.0, 200.0)))
    metadata = {"stitch_beat_count": 3, "metadata": {"qrs_trigger_times": np.arange(7, dtype=np.float64)}}

    prepared = prepare_3d_brightness_for_display(volumes, timestamps, metadata)

    assert prepared.was_stitched is True
    assert prepared.volumes.shape == (4, 6, 4, 4)
    np.testing.assert_allclose(prepared.timestamps, [0.05, 0.95, 3.05, 3.95])
    _assert_elevation_planes(prepared.volumes, [50.0, 100.0, 100.0, 150.0, 150.0, 200.0])


def test_prepare_3d_brightness_can_stitch_on_raw_timestamp_axis() -> None:
    volumes = np.arange(8, dtype=np.uint8).reshape(8, 1, 1, 1)
    timestamps = np.asarray([10.05, 10.95, 11.05, 11.95, 12.05, 12.95, 13.05, 13.95], dtype=np.float64)
    metadata = {
        "stitch_beat_count": 4,
        "metadata": {
            "time_reference": {"volume_time_origin_s": 10.0},
            "qrs_trigger_times": [10.0, 11.0, 12.0, 13.0, 14.0],
        },
    }

    prepared = prepare_3d_brightness_for_display(volumes, timestamps, metadata)

    assert prepared.was_stitched is True
    assert prepared.volumes.shape == (2, 4, 1, 1)
    assert np.allclose(prepared.timestamps, [0.05, 0.95])
    assert prepared.volumes[:, :, 0, 0].tolist() == [[6, 4, 2, 0], [7, 5, 3, 1]]


def test_prepare_3d_brightness_noop_stitch_keeps_relative_timeline() -> None:
    volumes = np.arange(4, dtype=np.uint8).reshape(4, 1, 1, 1)
    timestamps = np.asarray([10.1, 10.2, 10.3, 10.4], dtype=np.float64)
    metadata = {
        "stitch_beat_count": 3,
        "metadata": {
            "time_reference": {"volume_time_origin_s": 10.0},
            "qrs_trigger_times": [10.0, 11.0, 12.0, 13.0],
        },
    }

    prepared = prepare_3d_brightness_for_display(volumes, timestamps, metadata)

    assert prepared.was_stitched is False
    assert prepared.volumes.shape == volumes.shape
    assert np.allclose(prepared.timestamps, [0.1, 0.2, 0.3, 0.4])
    assert prepared.source_timestamps is None


def _synthetic_three_beat_windows(beat_planes: tuple[tuple[float, float], ...]) -> tuple[np.ndarray, np.ndarray]:
    timestamps = []
    volumes = []
    for window_start in (0, 3):
        for beat_index, planes in enumerate(beat_planes):
            for phase in (0.05, 0.95):
                timestamps.append(window_start + beat_index + phase)
                volumes.append(_constant_elevation_volume(planes))
    return np.stack(volumes, axis=0), np.asarray(timestamps, dtype=np.float64)


def _constant_elevation_volume(planes: tuple[float, float]) -> np.ndarray:
    volume = np.zeros((2, 4, 4), dtype=np.float32)
    volume[0] = planes[0]
    volume[1] = planes[1]
    return volume


def _assert_elevation_planes(volumes: np.ndarray, expected: list[float]) -> None:
    for frame in volumes:
        np.testing.assert_allclose(frame[:, 0, 0], expected)
