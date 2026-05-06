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
    resize_pixel_xy,
    sector_depth_ticks,
    sector_geometry_from_mapping,
    sector_to_cartesian,
    slice_volume,
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
    timestamps = np.asarray([10.1, 10.2, 11.1, 11.2, 12.1, 12.2, 13.1, 13.2], dtype=np.float64)
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
    assert np.allclose(prepared.timestamps, [0.1, 0.2])
    assert prepared.volumes[:, :, 0, 0].tolist() == [[0, 2, 4, 6], [1, 3, 5, 7]]


def test_prepare_3d_brightness_can_stitch_on_raw_timestamp_axis() -> None:
    volumes = np.arange(8, dtype=np.uint8).reshape(8, 1, 1, 1)
    timestamps = np.asarray([10.1, 10.2, 11.1, 11.2, 12.1, 12.2, 13.1, 13.2], dtype=np.float64)
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
    assert np.allclose(prepared.timestamps, [0.1, 0.2])
    assert prepared.volumes[:, :, 0, 0].tolist() == [[0, 2, 4, 6], [1, 3, 5, 7]]


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
