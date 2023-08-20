import warnings

import numpy as np
from magicgui import magic_factory
from napari_tools_menu import register_function
import napari
from napari.types import LayerDataTuple
from typing import List
from copy import deepcopy
from ._utils import compute_combined_slices

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


# This is the actual plugin function, where we export our function
# (The functions themselves are defined below)
## THIS FUNCTION BELOW WAS TAKEN FROM napari-crop. I do not own this code, I'm just trying to learn
## how Napari plugins work :)
@register_function(menu="Utilities > Crop region(s) (napari-crop)")
def crop_region(
    to_be_cropped: napari.layers.Layer,
    to_be_cropped_points: napari.layers.Points,
    cropping_layer: napari.layers.Shapes,
    viewer: 'napari.viewer.Viewer' = None,
) -> List[napari.types.LayerDataTuple]:
    """Crop regions in napari defined by shapes."""
    if cropping_layer is None:
        cropping_layer.mode = "add_rectangle"
        warnings.warn("Please annotate a region to crop.")
        return

    layer_data_image, layer_props_image, layer_type_image = to_be_cropped.as_layer_data_tuple()

    if (
        isinstance(to_be_cropped, napari.layers.Image)
        or isinstance(to_be_cropped, napari.layers.Labels)
    ):
        pass
    else:
        warnings.warn("Please select an image or labels to_be_cropped to crop.")

    try:
        rgb = layer_props_image["rgb"]
    except KeyError:
        rgb = False

    # Select points overlapping with polygon
    points = to_be_cropped_points.as_layer_data_tuple()[0]
    outPoly = []
    polygon = Polygon(cropping_layer.as_layer_data_tuple()[0][0])
    for i, po in enumerate(points):
        poo = Point(po[0], po[1])
        if not polygon.contains(poo):
            outPoly.append(i)
    cropped_points = deepcopy(to_be_cropped_points)
    cropped_points.selected_data = napari.utils.events.Selection(tuple(outPoly))
    cropped_points.remove_selected()

    layer_data_points, layer_props_points, layer_type_points = cropped_points.as_layer_data_tuple()
    
    shape_types = cropping_layer.shape_type
    shapes = cropping_layer.data
    cropped_list = []
    new_layer_index = 0
    new_name = layer_props_image["name"] + " cropped [0]"
    names_list = []
    if viewer is not None:
        # Get existing to_be_cropped names in viewer
        names_list = [to_be_cropped.name for to_be_cropped in viewer.layers]
    for shape_count, [shape, shape_type] in enumerate(zip(shapes,
                                                          shape_types)):
        # move shape vertices to within image coordinate limits
        layer_shape = np.array(layer_data_image.shape)
        if rgb:
            layer_shape = layer_shape[:-1]
        # find min and max for each dimension
        start = np.rint(np.min(shape, axis=0))
        stop = np.rint(np.max(shape, axis=0))
        # create slicing indices
        slices = tuple(
            slice(first, last + 1) if first != last else slice(0, None)
            for first, last in np.stack([start.clip(0),
                                         stop.clip(0)]).astype(int).T
        )
        cropped_data = layer_data_image[slices].copy()
        # handle polygons
        if shape_type != "rectangle":
            mask_nD_shape = np.array(
                [1 if slc.stop is None
                 else (min(slc.stop, layer_data_image.shape[i]) - slc.start)
                 for i, slc in enumerate(slices)]
            )
            # remove extra dimensions from shape vertices
            # (draw in a single plane)
            verts_flat = np.array(shape - start.clip(0))[:, mask_nD_shape > 1]
            # get a 2D mask
            mask_2D = (
                napari.layers.Shapes(verts_flat, shape_type=shape_type)
                .to_masks()
                .squeeze()
            )
            # match cropped_data (x,y) shape with mask_2D shape
            cropped_data_shape = cropped_data.shape
            # Adjust cropped_data axes order in case axes were swapped in napari
            if viewer is not None:
                cropped_data_shape = np.moveaxis(cropped_data,
                                                 viewer.dims.order,
                                                 np.arange(len(layer_shape))).shape
            if rgb:
                shape_dif_2D = np.array(cropped_data_shape[-3:-1]) \
                    - np.array(mask_2D.shape)
            else:
                shape_dif_2D = np.array(cropped_data_shape[-2:]) \
                    - np.array(mask_2D.shape)
            shape_dif_2D = [None if i == 0 else i
                            for i in shape_dif_2D.tolist()]
            mask_2D = mask_2D[:shape_dif_2D[-2], :shape_dif_2D[-1]]
            # add back the rgb(a) dimension
            if rgb:
                mask_nD_shape = np.append(mask_nD_shape, 1)
            # add back dimensions of the original vertices
            mask_nD = mask_2D.reshape(mask_nD_shape)
            # broadcast the mask to the shape of the cropped image
            mask = np.broadcast_to(mask_nD, cropped_data.shape)
            # erase pixels outside drawn shape
            cropped_data[~mask] = 0

            # trim zeros
            inner_slices = get_nonzero_slices(cropped_data)
            cropped_data = trim_zeros(cropped_data, rgb=rgb)
            slices = compute_combined_slices(layer_shape, slices, inner_slices)

        new_layer_props = layer_props_image.copy()
        # Update start and stop values for bbox
        start = [slc.start for slc in slices if slc is not None]
        stop = []
        for slc in slices:
            if slc is not None:
                if slc.stop is None:
                    stop.append(layer_shape[slices.index(slc)])
                else:
                    stop.append(slc.stop)
        # stop = [slc.stop for slc in slices if slc is not None]
        # Add cropped coordinates as metadata
        # bounding box: ([min_z,] min_row, min_col, [max_z,] max_row, max_col)
        # Pixels belonging to the bounding box are in the half-open interval [min_row; max_row) and [min_col; max_col).
        new_layer_props['metadata'] = {'bbox': tuple(start + stop)}
        # apply to_be_cropped translation scaled by to_be_cropped scaling factor
        new_layer_props['translate'] = tuple(np.asarray(tuple(start)) * np.asarray(layer_props_image['scale']))

        # If to_be_cropped name is in viewer or is about to be added,
        # increment to_be_cropped name until it has a different name
        while True:
            new_name = layer_props_image["name"] \
                + f" cropped [{shape_count+new_layer_index}]"
            if new_name not in names_list:
                break
            else:
                new_layer_index += 1
        new_layer_props["name"] = new_name
        names_list.append(new_name)
        cropped_list.append((cropped_data, new_layer_props, layer_type_image))
        cropped_list.append((layer_data_points, layer_props_points, layer_type_points))
    return cropped_list


def get_nonzero_slices(array):
    non_zero = np.where(array != 0)
    return [slice(min(i_nz), max(i_nz) + 1) for i_nz in non_zero]


def trim_zeros(image, rgb=False):
    slices = get_nonzero_slices(image)
    if rgb:
        slices[-1] = slice(None)
    return image[tuple(slices)]


def cut_with_plane(image_to_be_cut, plane_normal, plane_position, positive_cut=True, crop=False):
    """Cut a 3D volume with a plane

    Parameters
    ----------
    image_to_be_cut : array_like (3D)
        3D image to be cut
    plane_normal : tuple (3)
        Normal vector of the plane
    plane_position : tuple (3)
        Position of the plane center
    positive_cut : bool, optional
        If True, the positive side of the plane is kept.
        If False, the negative side of the plane is kept. By default True

    Returns
    -------
    array_like (3D)
        Cut image with the same shape as the input image
    """
    import numpy as np
    if len(image_to_be_cut.shape) != 3:
        print('Input image to be cut must be 3D')
        return
    image_to_be_cut = np.asarray(image_to_be_cut)

    x = np.arange(image_to_be_cut.shape[2])
    y = np.arange(image_to_be_cut.shape[1])
    z = np.arange(image_to_be_cut.shape[0])

    zmesh, ymesh, xmesh = np.meshgrid(z, y, x, indexing='ij')

    xm = xmesh - plane_position[2]
    ym = ymesh - plane_position[1]
    zm = zmesh - plane_position[0]

    p = xm * plane_normal[2] + ym * plane_normal[1] + zm * plane_normal[0]

    if positive_cut:
        mask = np.where(p >= 0, 1, 0).astype(bool)
    else:
        mask = np.where(p < 0, 1, 0).astype(bool)

    image_cut = image_to_be_cut.copy()
    image_cut[~mask] = 0
    if crop:
        image_cut = trim_zeros(image_cut)
    return image_cut
