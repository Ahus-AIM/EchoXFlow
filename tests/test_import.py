def test_imports_package() -> None:
    import echoxflow

    assert echoxflow.__doc__


def test_preview_entry_points_are_exported() -> None:
    from echoxflow import common_preview_arrays, write_preview_pair
    from echoxflow.plotting import blend_segmentation_rgb

    assert callable(common_preview_arrays)
    assert callable(write_preview_pair)
    assert callable(blend_segmentation_rgb)
