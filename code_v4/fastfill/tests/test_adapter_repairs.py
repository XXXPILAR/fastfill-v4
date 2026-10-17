"""Bounded adapter regressions; fixtures are self-contained, with no corpus/audit files read.

The 21 source boxes below are the complete retained >2 cm error examples in the
2026-09-27 adapter evidence (11 Structured3D, 8 HSSD200, 2 MultiScan).
"""
import copy
import csv
import io
import itertools
import json
import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from fastfill.adapters import hssd200, interiorgs, multiscan, structured3d
from fastfill.adapters.unified import convert_room
from fastfill.build import prep


AUDITED_BOXES = [('Structured3D',
  'structured3d_00160_room_584/50',
  {'category': 'unknown',
   'instance_id': 50,
   'position': {'x': 1.516548, 'y': 0.92628, 'z': 0.4325},
   'raw_basis': [[-0.9999999999999962, 8.742277644230737e-08, 0.0],
                 [3.8213709261743826e-15, 4.3711388286737763e-08, -0.9999999999999991],
                 [-8.742277644230729e-08, -0.9999999999999953, -4.3711388175715626e-08]],
   'raw_coeffs': [208.00000000000077, 32.50000365650003, 598.0000000000028]}),
 ('Structured3D',
  'structured3d_00393_room_584/67',
  {'category': 'unknown',
   'instance_id': 67,
   'position': {'x': 1.305684, 'y': 0.087636, 'z': 0.4325},
   'raw_basis': [[-0.9999999999999962, 8.742278110047815e-08, 0.0],
                 [3.821371129789495e-15, 4.3711388286737763e-08, -0.9999999999999991],
                 [-8.742278110047807e-08, -0.9999999999999953, -4.3711388175715626e-08]],
   'raw_coeffs': [209.00000000000077, 32.50000365650003, 548.0000000000027]}),
 ('Structured3D',
  'structured3d_01159_room_232/100',
  {'category': 'table',
   'instance_id': 100,
   'position': {'x': 0.44347, 'y': -2.189658, 'z': 0.3155},
   'raw_basis': [[4.371139006309477e-08, 3.821370999999997e-15, -0.9999999999999991],
                 [8.742277999999968e-08, -0.999999999999996, 2.2499312661442353e-22],
                 [-0.9999999999999953, -8.742277999999959e-08, -4.3711389952072466e-08]],
   'raw_coeffs': [35.00000000000003, 18.75000000000007, 12.452050000000058]}),
 ('Structured3D',
  'structured3d_02687_room_670/53',
  {'category': 'unknown',
   'instance_id': 53,
   'position': {'x': -0.730234, 'y': -1.066933, 'z': 0.3145},
   'raw_basis': [[1.0, 0.0, 0.0],
                 [0.0, -4.371138828673793e-08, -0.9999999999999991],
                 [-0.0, 0.9999999999999991, -4.3711388175715626e-08]],
   'raw_coeffs': [223.5, 32.50000365650003, 336.00000000000034]}),
 ('Structured3D',
  'structured3d_02687_room_670/54',
  {'category': 'unknown',
   'instance_id': 54,
   'position': {'x': -0.330234, 'y': -1.066933, 'z': 0.3145},
   'raw_basis': [[1.0, 0.0, 0.0],
                 [0.0, -4.371138828673793e-08, -0.9999999999999991],
                 [-0.0, 0.9999999999999991, -4.3711388175715626e-08]],
   'raw_coeffs': [173.5, 32.50000365650003, 336.00000000000034]}),
 ('Structured3D',
  'structured3d_02687_room_670/55',
  {'category': 'unknown',
   'instance_id': 55,
   'position': {'x': 0.019766, 'y': -1.066933, 'z': 0.3145},
   'raw_basis': [[1.0, 0.0, 0.0],
                 [0.0, -4.371138828673793e-08, -0.9999999999999991],
                 [-0.0, 0.9999999999999991, -4.3711388175715626e-08]],
   'raw_coeffs': [173.5, 32.50000365650003, 336.00000000000034]}),
 ('Structured3D',
  'structured3d_03087_room_332351/56',
  {'category': 'bed',
   'instance_id': 56,
   'position': {'x': 1.063394, 'y': 1.34462, 'z': 0.3805},
   'raw_basis': [[1.9984014443252818e-15, 4.371138999999996e-08, -0.9999999999999991],
                 [0.9999999999999991, -4.37113899041451e-08, 0.0],
                 [-4.3711389793122756e-08, -0.9999999999999981, -4.371139e-08]],
   'raw_coeffs': [90.66900000000008, 12.37120000000001, 15.869400000000033]}),
 ('Structured3D',
  'structured3d_03155_room_309/16',
  {'category': 'table',
   'instance_id': 16,
   'position': {'x': -0.60826, 'y': 4.415584, 'z': 0.2747},
   'raw_basis': [[-3.3306690738754696e-15, -4.371138999999996e-08, -0.9999999999999991],
                 [-0.9999999999999972, 7.549789999487077e-08, -1.1102230246251565e-16],
                 [7.549789999487071e-08, 0.9999999999999962, -4.3711390000000065e-08]],
   'raw_coeffs': [35.00000000000003, 18.750000000000053, 12.452050000000046]}),
 ('Structured3D',
  'structured3d_03356_room_584/129',
  {'category': 'table',
   'instance_id': 129,
   'position': {'x': 2.124656, 'y': -1.657159, 'z': 0.4325},
   'raw_basis': [[-4.371139117331779e-08, 0.9999999999999992, 0.0],
                 [4.3711388286737896e-08, 1.9106855776612683e-15, -0.9999999999999991],
                 [-0.9999999999999983, -4.3711390840250886e-08, -4.3711388175715626e-08]],
   'raw_coeffs': [209.0000000000002, 32.50000365650003, 273.0000000000005]}),
 ('Structured3D',
  'structured3d_03356_room_584/133',
  {'category': 'unknown',
   'instance_id': 133,
   'position': {'x': 1.409656, 'y': -0.555107, 'z': 0.4325},
   'raw_basis': [[-4.371139117331779e-08, -0.9999999999999992, 0.0],
                 [-4.3711388286737896e-08, 1.9106855776612683e-15, -0.9999999999999991],
                 [0.9999999999999983, -4.3711390840250886e-08, -4.3711388175715626e-08]],
   'raw_coeffs': [209.0000000000002, 32.50000365650003, 438.00000000000097]}),
 ('Structured3D',
  'structured3d_03356_room_584/139',
  {'category': 'unknown',
   'instance_id': 139,
   'position': {'x': 1.409656, 'y': -1.657159, 'z': 0.4325},
   'raw_basis': [[-4.371139117331779e-08, 0.9999999999999992, 0.0],
                 [4.3711388286737896e-08, 1.9106855776612683e-15, -0.9999999999999991],
                 [-0.9999999999999983, -4.3711390840250886e-08, -4.3711388175715626e-08]],
   'raw_coeffs': [209.0000000000002, 32.50000365650003, 438.00000000000097]}),
 ('HSSD200',
  'HSSD200:102816627:3/102816627:267',
  {'category': 'book',
   'scene_id': '102816627',
   'instance_id': '267',
   'obb_center': '[3.546474, 0.121647, -1.923595]',
   'obb_half_extents': '[0.047279, 0.12895, 0.102077]',
   'obb_rotation_wxyz': '[0.65328163, 0.6532815, -0.27059783, -0.27059788]'}),
 ('HSSD200',
  'HSSD200:102816627:3/102816627:309',
  {'category': 'book',
   'scene_id': '102816627',
   'instance_id': '309',
   'obb_center': '[3.546474, 1.921797, -1.923595]',
   'obb_half_extents': '[0.047279, 0.12895, 0.102077]',
   'obb_rotation_wxyz': '[0.65328163, 0.6532815, -0.27059783, -0.27059788]'}),
 ('HSSD200',
  'HSSD200:103997781_171030978:3/103997781_171030978:185',
  {'category': 'book',
   'scene_id': '103997781_171030978',
   'instance_id': '185',
   'obb_center': '[-2.296859, 0.115267, -4.154386]',
   'obb_half_extents': '[0.044784, 0.12895, 0.09669]',
   'obb_rotation_wxyz': '[0.65407459, 0.65248737, -0.27139045, -0.26980341]'}),
 ('HSSD200',
  'HSSD200:104348082_171512994:0/104348082_171512994:150',
  {'category': 'book',
   'scene_id': '104348082_171512994',
   'instance_id': '150',
   'obb_center': '[-9.489737, 0.149516, -1.599131]',
   'obb_half_extents': '[0.0562, 0.104913, 0.121338]',
   'obb_rotation_wxyz': '[0.60044965, 0.31731935, 0.21637841, -0.70139078]'}),
 ('HSSD200',
  'HSSD200:104862501_172226556:5/104862501_172226556:165',
  {'category': 'book',
   'scene_id': '104862501_172226556',
   'instance_id': '165',
   'obb_center': '[-4.624166, 0.121647, 4.894016]',
   'obb_half_extents': '[0.047279, 0.12895, 0.102077]',
   'obb_rotation_wxyz': '[0.65328142, 0.2705982, 0.27059834, -0.65328136]'}),
 ('HSSD200',
  'HSSD200:104862501_172226556:5/104862501_172226556:207',
  {'category': 'book',
   'scene_id': '104862501_172226556',
   'instance_id': '207',
   'obb_center': '[-3.159476, 0.121647, -0.588035]',
   'obb_half_extents': '[0.047279, 0.12895, 0.102077]',
   'obb_rotation_wxyz': '[0.65328163, 0.6532815, -0.27059783, -0.27059788]'}),
 ('HSSD200',
  'HSSD200:106879005_174887124:4/106879005_174887124:98',
  {'category': 'roof',
   'scene_id': '106879005_174887124',
   'instance_id': '98',
   'obb_center': '[-5.67073, 3.956208, -6.830123]',
   'obb_half_extents': '[6.5, 0.55117, 2.749985]',
   'obb_rotation_wxyz': '[0.97029571, -0.24192195, 0.0, 0.0]'}),
 ('HSSD200',
  'HSSD200:107734158_175999998:14/107734158_175999998:448',
  {'category': 'book',
   'scene_id': '107734158_175999998',
   'instance_id': '448',
   'obb_center': '[-9.930676, 0.157266, -2.633861]',
   'obb_half_extents': '[0.060269, 0.12895, 0.130123]',
   'obb_rotation_wxyz': '[-0.21313372, -0.70393206, 0.59736535, -0.31970036]'}),
 ('MultiScan',
  'MultiScan::scene_00029_01/o43',
  {'category': 'bag',
   'object_id': '43',
   'obb_center': '[-2.216006052445697, -3.198990954172932, -2.0838447429477776]',
   'obb_axes': '[-0.9391466459853527, 0.34151555401180256, -0.03702301584790649, 0.2533428178007576, '
               '0.6157978529307632, -0.7460632821648807, -0.23199352146824048, -0.7100423442963621, '
               '-0.6648450009610445]',
   'obb_half_extents': '[0.2190989368345001, 0.2549238904394131, 0.1397719718865565]',
   'front': '[0.23199352146824054, 0.7100423442963623, 0.6648450009610446]',
   'is_architectural': 'False',
   'is_opening': 'False'}),
 ('MultiScan',
  'MultiScan::scene_00109_04/o31',
  {'category': 'backpack',
   'object_id': '31',
   'obb_center': '[0.6333183612225775, 1.7997251504673377, -0.9412570338535274]',
   'obb_axes': '[-0.9961946980917455, -0.022557566113149824, -0.0841859828293692, -0.08715574274765812, '
               '0.25783416049629937, 0.9622501868990583, -2.7755575615628914e-17, 0.9659258262890683, '
               '-0.2588190451025206]',
   'obb_half_extents': '[0.19922853225692852, 0.20651165445598438, 0.11010368672622178]',
   'front': '[2.7755575615628914e-17, -0.9659258262890683, 0.2588190451025206]',
   'is_architectural': 'False',
   'is_opening': 'False'})]


def rotation(yaw, roll=0, pitch=0):
    c, s = math.cos(yaw), math.sin(yaw)
    a, b = math.cos(roll), math.sin(roll)
    d, e = math.cos(pitch), math.sin(pitch)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.array(
        [[1, 0, 0], [0, a, -b], [0, b, a]]) @ np.array([[d, 0, e], [0, 1, 0], [-e, 0, d]])


def quaternion_matrix(q):
    # Independent vector/quaternion formula, applied to the three basis vectors.
    w, v = q[0], np.asarray(q[1:])
    return np.column_stack([e + 2 * np.cross(v, w * e + np.cross(v, e)) for e in np.eye(3)])


def corners(R, size, center):
    return np.asarray(list(itertools.product((-0.5, 0.5), repeat=3))) * size @ R.T + center


def pose(f):
    return (np.array([f['furniture_size'][k] for k in ('width', 'length', 'height')]),
            np.array([f['furniture_position'][k] for k in 'xyz']),
            math.radians(f['furniture_rotation']['z']))


def assert_envelope(f, points, center_z):
    size, pos, yaw = pose(f)
    local = (points - pos) @ rotation(yaw)
    expected_lo = -size / 2 if center_z else np.array([-size[0] / 2, -size[1] / 2, 0])
    np.testing.assert_allclose(local.min(0), expected_lo, atol=1e-8)
    np.testing.assert_allclose(local.max(0), expected_lo + size, atol=1e-8)


@pytest.mark.parametrize('source,uid,raw', AUDITED_BOXES, ids=[r[1] for r in AUDITED_BOXES])
def test_all_audited_retained_boxes_enclose_rotated_corners(source, uid, raw):
    before = copy.deepcopy(raw)
    if source == 'Structured3D':
        f = structured3d._furniture(raw)
        R, size = np.asarray(raw['raw_basis']).T, 2 * np.abs(raw['raw_coeffs']) / 1000
        center = np.array([raw['position'][k] for k in 'xyz'])
    elif source == 'HSSD200':
        f = hssd200._furniture(raw, 0)
        R = hssd200.T @ quaternion_matrix(json.loads(raw['obb_rotation_wxyz']))
        size = 2 * np.asarray(json.loads(raw['obb_half_extents']))
        center = hssd200.T @ np.asarray(json.loads(raw['obb_center']))
    else:
        f = multiscan.box(raw)
        R = np.asarray(json.loads(raw['obb_axes'])).reshape(3, 3).T
        size = 2 * np.asarray(json.loads(raw['obb_half_extents']))
        center = np.asarray(json.loads(raw['obb_center']))
    assert_envelope(f, corners(R, size, center), source != 'MultiScan')
    assert raw == before


def source_row(source, R, size, center, category='table', identity='1'):
    """Encode one semantic-front X / up Z box in each source's actual axis convention."""
    if source == 'Structured3D':
        return {'category': category, 'instance_id': identity, 'position': dict(zip('xyz', center)),
                'raw_basis': [R[:, 1].tolist(), (-R[:, 0]).tolist(), R[:, 2].tolist()],
                'raw_coeffs': (np.asarray(size)[[1, 0, 2]] * 500).tolist()}
    C = R[:, [1, 2, 0]]  # asset +Z front, +Y up
    row = {'category': category, 'obb_half_extents': json.dumps((np.asarray(size)[[1, 2, 0]] / 2).tolist())}
    if source == 'HSSD200':
        M = hssd200.T.T @ C
        w = math.sqrt(1 + np.trace(M)) / 2
        q = [w, (M[2, 1] - M[1, 2]) / (4 * w), (M[0, 2] - M[2, 0]) / (4 * w),
             (M[1, 0] - M[0, 1]) / (4 * w)]
        return {**row, 'scene_id': 'synthetic', 'instance_id': identity,
                'obb_center': json.dumps((hssd200.T.T @ center).tolist()), 'obb_rotation_wxyz': json.dumps(q)}
    return {**row, 'object_id': identity, 'obb_center': json.dumps(list(center)),
            'obb_axes': json.dumps(C.T.flatten().tolist()), 'front': json.dumps(R[:, 0].tolist()),
            'is_architectural': 'False', 'is_opening': 'False'}


def furniture(source, row, floor=0):
    if source == 'Structured3D':
        return structured3d._furniture(row)
    if source == 'HSSD200':
        return hssd200._furniture(row, floor)
    return multiscan.box(row)


@pytest.mark.parametrize('source', ['Structured3D', 'HSSD200', 'MultiScan'])
@pytest.mark.parametrize('yaw', [-1.2, 0.7, 2.1])
def test_tilt_preserves_horizontal_front_and_exact_center(source, yaw):
    R, size, center = rotation(yaw, 0.4), np.array([1.4, 0.8, 0.5]), np.array([10., 20., 2.])
    f = furniture(source, source_row(source, R, size, center))
    assert_envelope(f, corners(R, size, center), source != 'MultiScan')
    assert math.cos(pose(f)[2] - yaw) == pytest.approx(1)
    assert f['furniture_rotation']['x'] != 0
    assert f.get('front_known') is not False


@pytest.mark.parametrize('source', ['Structured3D', 'HSSD200', 'MultiScan'])
def test_upright_known_front_unchanged(source):
    R, size, center = rotation(0.7), np.array([1.4, 0.8, 0.5]), np.array([10., 20., 2.])
    f = furniture(source, source_row(source, R, size, center))
    assert_envelope(f, corners(R, size, center), source != 'MultiScan')
    np.testing.assert_allclose(pose(f)[0], size)
    assert pose(f)[2] == pytest.approx(0.7)
    assert f['furniture_rotation']['x'] == 0


@pytest.mark.parametrize('source', ['Structured3D', 'MultiScan'])
def test_vertical_front_has_geometry_but_no_claimed_heading(source):
    R = rotation(0.7, pitch=math.pi / 2)
    size, center = np.array([1.4, 0.8, 0.5]), np.array([10., 20., 2.])
    f = furniture(source, source_row(source, R, size, center))
    assert_envelope(f, corners(R, size, center), source != 'MultiScan')
    assert f.get('front_known') is False or f.get('front_vertical') is True


ARGS = SimpleNamespace(boundary_types=['polygon', 'hull'], source_anchors={}, anchors=['floor', 'object'],
                       min_objects=1, max_vertices=0, oob_tol=.1, hidden_max=.3, reject_flagged=[])


def test_hssd_floor_origin_and_support_use_true_envelope():
    R, size, floor = rotation(.7, .4), np.array([1.4, .8, .5]), 7.25
    height = np.ptp(corners(R, size, np.zeros(3))[:, 2])
    table = hssd200._furniture(source_row('HSSD200', R, size, np.array([12., 22., floor + height / 2])), floor)
    book = hssd200._furniture(source_row('HSSD200', rotation(.7), np.array([.2, .2, .1]),
                                       np.array([12., 22., floor + height + .05]), 'book', '2'), floor)
    room = {'room_id': 'r', 'room_boundary': [[10, 20], [15, 20], [15, 25], [10, 25]],
            'room_height': 3, 'furniture': [table, book]}
    ir = convert_room(room, source='HSSD200', uid='r', group='g', boundary_type='polygon', center_z=True,
                      meta={'front_known': True})
    assert ir['objects'][0]['pos'] == pytest.approx([2, 2, 0])
    built, why = prep(ir, ARGS)
    assert why is None
    child = next(o for o in built['objects'] if o['id'] == 'synthetic:2')
    assert child['anchor'] == 'object' and child['parent'] == 'synthetic:1'
    assert child['pos'][2] == round(height, 2)


def test_structured_false_floor_obstacle_is_removed_after_preparation():
    raw = AUDITED_BOXES[0][2]  # structured3d_00160_room_584 / instance 50
    chair = source_row('Structured3D', rotation(0), np.array([.6, .6, 1.]),
                       np.array([6., 6., .5]), 'chair', 'control')
    record = {'sample_id': 'bounded', 'provenance': {'source_house_id': 'test', 'split': 'train'},
              'room_origin_house_m': [0, 0, 0],
              'room_context': {'room_id': 'r', 'room_type': 'office', 'ceiling_height_m': 3,
                               'floor_polygon': [[0, 0], [10, 0], [10, 10], [0, 10]]},
              'layout': {'floor_layout': {'objects': [raw, chair]}}}
    ir = structured3d.convert(record)
    false_obstacle = ir['objects'][0]
    assert false_obstacle['pos'][2] == pytest.approx(.4, abs=1e-6)
    assert false_obstacle['size'][2] == pytest.approx(.065, abs=1e-6)
    assert false_obstacle['front_known'] is False
    built, why = prep(ir, ARGS)
    assert why is None
    assert [o['id'] for o in built['objects']] == ['control']
    assert not built['fixed']


def csv_stream(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    stream.seek(0)
    return stream


def test_hssd_load_keeps_tilt_envelope_floor_and_front_metadata():
    floor = 7.25
    R, size = rotation(.7, .4), np.array([1.4, .8, .5])
    height = np.ptp(corners(R, size, np.zeros(3))[:, 2])
    center = np.array([12., 22., floor + height / 2])
    raw = source_row('HSSD200', R, size, center)
    row = {**raw, 'instance_kind': 'rigid', 'bbox_source': 'mesh', 'region_id': '0',
           'region_source': 'inside', 'is_architectural': 'False',
           'aabb_max': json.dumps((corners(R, size, center) @ hssd200.T).max(0).tolist())}
    region = {'scene_id': 'synthetic', 'region_id': '0', 'room_type': 'office', 'region_name': 'test',
              'poly_loop_xz': '[[10,-20],[15,-20],[15,-25],[10,-25]]',
              'floor_height': floor, 'ceiling_height': floor + 3, 'extrusion_height': 3}
    files = {'objects.csv': [row], 'regions.csv': [region],
             'asset_catalog.csv': [{'template_name': 'unused', 'local_bbox_min': ''}]}
    with patch('builtins.open', side_effect=lambda path: csv_stream(files[path.rsplit('/', 1)[-1]])):
        ir, = hssd200.load('unused')
    obj, = ir['objects']
    assert obj['pos'] == pytest.approx([2, 2, 0])
    assert obj['yaw'] == pytest.approx(.7)
    assert obj['size'][2] == pytest.approx(height)
    assert obj['tilted'] and 'rot_wxyz_yup' in obj
    assert ir['meta']['floor_height'] == floor


def test_multiscan_load_uses_source_floor_and_propagates_unknown_front():
    floor, R, size = -3.25, rotation(.7, pitch=math.pi / 2), np.array([.8, .6, .4])
    row = {**source_row('MultiScan', R, size, np.array([12., 22., floor + .4])),
           'scan_id': 'scene_9_0', 'aabb_min': json.dumps([11, 21, floor])}
    files = {'objects.csv': [row], 'relations.csv': [{'scan_id': 'scene_9_0', 'subject_id': '1',
                                                    'relation': 'rests_on_floor'}],
             'scans.csv': [{'scan_id': 'scene_9_0', 'scene_id': 'scene_9', 'room_type': 'office', 'device': 'test'}],
             'regions.csv': [{'scan_id': 'scene_9_0', 'floor_height': floor + .2, 'ceiling_height': floor + 3,
                              'height_reliable': 'True', 'poly_source': 'floor', 'wall_segments': '[]',
                              'poly_loop': '[[10,20],[15,20],[15,25],[10,25]]'}]}
    with patch('builtins.open', side_effect=lambda path: csv_stream(files[path.rsplit('/', 1)[-1]])):
        ir, = multiscan.load('unused')
    assert ir['height'] == 3
    assert ir['meta']['floor_z'] == floor
    assert ir['objects'][0]['pos'] == pytest.approx([2, 2, 0])
    assert ir['objects'][0].get('front_known') is False


def interior_box(identity, xy, category='chair'):
    pts = [[xy[0] + x, xy[1] + y, z] for z in (0., 1.)
           for x, y in ((.2, -.2), (-.2, -.2), (-.2, .2), (.2, .2))]
    return {'furniture_category': category, 'furniture_instance_id': identity,
            'furniture_position': {'x': 999, 'y': 999, 'z': 999},
            'source_fields': {'bounding_box': [dict(zip('xyz', p)) for p in pts]}}


def interior_room(identity, ring, furn=()):
    return {'room_id': identity, 'room_boundary': ring, 'room_height': 3, 'furniture': list(furn)}


def load_interior(scene):
    with patch.object(interiorgs, 'iter_records', return_value=iter([scene])):
        return list(interiorgs.load('unused'))


def test_interior_recovers_only_unique_bbox_centers_without_mutating_assignments():
    # Repeated closing/zero-length edges triggered the upstream ray-cast defect.
    left = [[10, 20], [12, 20], [12, 22], [10, 22], [10, 22], [10, 20]]
    right = [[12, 20], [14, 20], [14, 22], [12, 22]]
    original = interior_box('already', [11, 21])
    candidates = [interior_box('left', [11.5, 21]), interior_box('right', [13, 21]),
                  interior_box('shared', [12, 21]), interior_box('outside', [100, 100]),
                  interior_box('already', [13, 21]),
                  {**interior_box('invalid', [11, 21]), 'source_fields': {'bounding_box': []}}]
    scene = {'scene_id': 'arbitrary-house', 'rooms': [interior_room('west', left, [original]),
                                                    interior_room('east', right)], 'unassigned_objects': candidates}
    before = copy.deepcopy(scene)
    rooms = load_interior(scene)
    assert [[o['id'] for o in r['objects']] for r in rooms] == [['already', 'left'], ['right']]
    assert rooms[0]['objects'][1]['pos'] == pytest.approx([1.5, 1, 0])
    recovery = rooms[0]['meta']['membership_recovery']
    assert recovery['scene_recovered'] == 2
    assert recovery['scene_unresolved'] == {'ambiguous': ['shared'], 'outside': ['outside'], 'invalid_bbox': ['invalid']}
    assert recovery['scene_already_assigned'] == ['already']
    assert scene == before
    assert all(prep(room, ARGS)[1] is None for room in rooms)


def test_interior_does_not_pick_first_overlapping_room_or_nearest_room():
    ring = [[0, 0], [4, 0], [4, 4], [0, 4]]
    scene = {'scene_id': 'overlap', 'rooms': [interior_room('a', ring), interior_room('b', ring)],
             'unassigned_objects': [interior_box('both', [1, 1]), interior_box('neither', [5, 5])]}
    for r in load_interior(scene):
        assert not r['objects']
        assert r['meta']['membership_recovery']['scene_unresolved'] == {
            'ambiguous': ['both'], 'outside': ['neither']}


@pytest.mark.parametrize('bad', [None, [], [[0, 0], [1, 1], [1, 0], [0, 1]]])
def test_interior_invalid_boundary_prevents_unproven_unique_assignment(bad):
    scene = {'scene_id': 'bad-ring', 'rooms': [interior_room('valid', [[0, 0], [4, 0], [4, 4], [0, 4]]),
                                              interior_room('invalid', bad)],
             'unassigned_objects': [interior_box('uncertain', [1, 1])]}
    for r in load_interior(scene):
        assert not r['objects']
        assert r['meta']['membership_recovery']['scene_unresolved'] == {'invalid_boundary': ['uncertain']}


def test_interior_duplicate_unassigned_ids_stay_unresolved():
    scene = {'scene_id': 'duplicate', 'rooms': [interior_room('a', [[0, 0], [4, 0], [4, 4], [0, 4]])],
             'unassigned_objects': [interior_box('repeat', [1, 1]), interior_box('repeat', [2, 2])]}
    r, = load_interior(scene)
    assert not r['objects']
    assert r['meta']['membership_recovery']['scene_unresolved'] == {'duplicate_id': ['repeat', 'repeat']}


def test_interior_assigned_only_scene_is_unchanged():
    room = interior_room('a', [[10, 20], [14, 20], [14, 24], [10, 24]], [interior_box('kept', [12, 22])])
    scene = {'scene_id': 'stable', 'rooms': [room]}
    ir, = load_interior(scene)
    assert ir == load_interior({**scene, 'unassigned_objects': []})[0]
    assert 'membership_recovery' not in ir['meta']
    assert ir['objects'][0]['pos'] == pytest.approx([2, 2, 0])
    assert len(ir['objects']) == 1
