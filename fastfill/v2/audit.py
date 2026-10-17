"""Read-only, bounded inventory and documented geometry semantics of local sources.

Inventory is not a full numerical/mesh audit of every large dataset. CSV counts
are measured; supplied manifests and README totals retain their evidence label.
Only MultiScan's admitted annotations undergo full per-row geometry validation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

TOP_DIRECTORIES = ("BillLin66__3D_Room_Collections", "imChuling__3D_Room_Collections",
                   "liantian__3D_Room_Scene_Collections", "XXXpilar__hssd_clean",
                   "XXXpilar__multiscan-clean")
BILL, IM, SAGE = TOP_DIRECTORIES[:3]
CANONICAL_SOURCE_ROOT = Path("/Volumes/harddisk/3D_Room_Collections")


def _source(name, directory, sample_file, **fields):
    return {"name": name, "directory": directory, "sample_file": sample_file,
            "category_description": "source category; no trustworthy scene-language assumed",
            "object_asset_id": "instance ID retained in provenance only; asset IDs never enter main condition",
            "support": "unknown unless explicit source metadata", "relations": "not used as condition by default",
            "semantic_front": "not independently established for all assets",
            "eligibility": "audit_only", **fields}


def source_catalog():
    """Versioned semantics, established from local raw records and export code."""
    return [
        _source("3D-FRONT (BillLin66)", f"{BILL}/3DFront_exported", "3dfront.jsonl",
                room_geometry="floor mesh convex hull; planar floor/ceiling height; openings",
                room_type="source room.type", size_semantics="asset-definition local size/bbox times instance scale",
                extent_kind="full; source bbox variants require independent verification", units="meters",
                position_reference="asset pivot/origin, not proven bottom/center", rotation="raw quaternion retained in houses; export yaw degrees",
                up_handedness="source Y-up -> (x,-z,y), right-handed Z-up", scale="export already applies abs(instance scale)",
                meshes="room structure vertices retained; furniture mesh not included",
                blockers="pivot-to-bbox offset and semantic front missing; many category/size nulls",
                declared_counts={"rooms": 56129, "objects": 533837},
                code_evidence=["kits/convert_3dfront.py"]),
        _source("SpatialGen", f"{BILL}/SpatialGen_exported", "spatialgen.jsonl",
                room_geometry="floor line loop; known self-intersection scene_00019", room_type="caption-derived mapping",
                size_semantics="instance-scaled local XYZ full extents", extent_kind="full", units="bbox meters; floor/ceiling millimeters",
                position_reference="box geometric center", rotation="full transform, mirrored/tilted rows retained",
                up_handedness="right-handed Z-up; reflection separately recorded", scale="norm of transform axis columns, once",
                meshes="not present", blockers="official test-only; semantic front must be established; reject tilt/mirrors and invalid rooms",
                declared_counts={"scenes": 48, "objects": 977}, code_evidence=["kits/convert_spatialgen.py"]),
        _source("3RScan", f"{IM}/3RScan_exported", "3rscan.json",
                room_geometry="no true room polygon/height; scan container", room_type="missing",
                size_semantics="obb.axesLengths along raw normalizedAxes", extent_kind="full", units="source metric OBBs, no export normalization",
                position_reference="obb.centroid", rotation="raw normalizedAxes; exported yaw null",
                up_handedness="scan frame, source axes need per-record verification", scale="OBB already observed metric geometry",
                support="3DSSG attributes/affordances", relations="3DSSG relationships preserved",
                meshes="not present", blockers="room reconstruction and semantic-front evidence required",
                declared_counts={"scans": 1482, "objects": 43450}, code_evidence=["kits/extract_3rscan.py"]),
        _source("ARKitScenes", f"{IM}/ARKitScenes_exported", "arkitscenes.json",
                room_geometry="no true room boundary/height", room_type="missing",
                size_semantics="segments.obbAligned.axesLengths local OBB", extent_kind="full", units="meters for obbAligned; raw obb is different",
                position_reference="export world AABB bottom; raw obbAligned.centroid is OBB center",
                rotation="normalizedAxes.T; raw orientation includes tilt", up_handedness="right-handed Z-up aligned frame",
                scale="aligned OBB metric extents; no asset scaling", meshes="not present; RGB keyframes archive only",
                object_asset_id="modelId custom.abandon is a placeholder, never a real CAD identity",
                blockers="front/room missing; official visit_id has one cross-split location group",
                declared_counts={"scans": 5047, "objects": 56605}, code_evidence=["extract_arkitscenes_json.py"]),
        _source("IL3D (3D-FRONT/HSSD/synthetic)", f"{IM}/IL3D_exported", "3dfront.json",
                room_geometry="source floor mesh; partial 3D-FRONT height/opening supplement", room_type="source category",
                size_semantics="layout bbox XYZ mapped to local X/Z/Y; verify asset metadata/model orientation",
                extent_kind="full", units="meters; USDA output separately multiplies by 100",
                position_reference="claimed bottom center; requires per-asset pivot verification",
                rotation="USD rotateXYZ matrix -> Z-up; complete matrix kept", up_handedness="right-handed Y-up -> (x,-z,y)",
                scale="source bbox vs scale requires asset audit; do not reapply automatically",
                support="placement_flags available; not mesh support surfaces", meshes="CAD USDZ asset archives not downloaded",
                blockers="semantic-front/pivot/asset axis audit; source NaNs and extreme outlier; opening world-span defect in v1",
                declared_counts={"scenes": 27816, "objects": 226978}, code_evidence=["aggregate_scenes.py"]),
        _source("InteriorGS", f"{IM}/InteriorGS_exported", "interiorgs.json",
                room_geometry="structure room profiles and holes", room_type="missing",
                size_semantics="long/short horizontal dimensions recovered from source 8 bbox corners",
                extent_kind="full", units="meters", position_reference="bbox bottom center",
                rotation="geometric long-axis angle modulo pi, no semantic front", up_handedness="right-handed Z-up; X right/Y back",
                scale="observed corners already metric", meshes="not present",
                blockers="geometry-only yaw cannot become semantic-front label; zero extents and ambiguous room assignments",
                declared_counts={"scenes": 1000, "rooms": 6607, "objects": 506061}, code_evidence=["aggregate_scenes.py"]),
        _source("InternScenes (Real2Sim/Gen)", f"{IM}/InternScenes_exported", "internscenes_gen_exported.json",
                room_geometry="floor convex hull; Real2Sim scan/region containers, Gen region labels", room_type="partial",
                size_semantics="source bbox[3:6] local extents before full ZXY rotation", extent_kind="full", units="meters",
                position_reference="bbox geometric center", rotation="bbox angle_z/x/y radians; ZXY order",
                up_handedness="right-handed Z-up layout; floor mesh Y-up converted", scale="box sizes already instance geometry",
                meshes="floor geometry only; full asset library absent", blockers="semantic front unverified; v1 geometric fallback must not become a front label",
                declared_counts={"scenes": 23672, "objects": 798954}, code_evidence=["kits/extract_internscenes_json.py"]),
        _source("MansionWorld", f"{IM}/MansionWorld_exported", "mansionworld.jsonl",
                room_geometry="room floor polygon, height, doors/windows from THOR source", room_type="source roomType",
                size_semantics="Objathor annotation bounding box; floor/wall/surface paths differ",
                extent_kind="full", units="meters", position_reference="floor/wall bottom center; surface path must inspect raw parent geometry",
                rotation="source THOR yaw mapped to output degrees", up_handedness="THOR Y-up -> (x,-z,y), right-handed Z-up",
                scale="metadata bbox; any asset scale must be checked against raw annotation",
                support="surface_groups with source parent associations", meshes="metadata/annotations only, no glTF/SDF",
                blockers="per-asset canonical front/pivot and support surfaces require audit; room/asset descriptions may leak target geometry",
                declared_counts={"buildings": 1000, "rooms": 27884, "objects": 405253}, code_evidence=["kits/convert_mansionworld.py"]),
        _source("OptiScene / 3D-SynthPlace", f"{IM}/OptiScene_exported", "holodeck.json",
                room_geometry="source rectangular proxy, no original floor polygon", room_type="source labels",
                size_semantics="bbox [height,width,depth] claimed local geometry; mesh verification missing",
                extent_kind="full", units="bbox centimeters; positions meters", position_reference="source_position_anchor, pivot semantics not established",
                rotation="source rotation.y is horizontal angle; full Euler order unspecified",
                up_handedness="source Y-up; export (x,z,y) reverses handedness; needs explicit yaw correction",
                scale="bbox source geometry; no unique future mesh implied", meshes="JSON-only export; scenes.tar.gz and prompts retained",
                blockers="handedness/yaw/pivot and known invalid geometry; do not reuse prompt bbox as v2 condition",
                declared_counts={"scenes": 17528, "objects": 147222}, code_evidence=["extract_optiscene_json.py", "OptiScene/gprompt.py"]),
        _source("SceneCAD / Scan2CAD", f"{IM}/SceneCAD_Scan2CAD_exported", "scenecad_scan2cad_combined.json",
                room_geometry="SceneCAD shell; scan may have multiple rooms; no true room partition", room_type="ScanNet sceneType",
                size_semantics="ScanNet objects world AABB; separate CAD local bbox half-extents × TRS scale",
                extent_kind="AABB full; raw CAD bbox half", units="meters", position_reference="ScanNet AABB bottom; CAD geometric OBB center",
                rotation="ScanNet yaw unavailable; CAD full transform incl axisAlignment", up_handedness="ShapeNet Y-up -> right-handed Z-up",
                scale="CAD 2*bbox*scale; no rotated AABB used as local size", meshes="not present",
                blockers="v1 tilted-CAD defect; semantic front and room grouping need raw-source adapter",
                declared_counts={"scenes": 1509, "cad_scenes": 1506, "cad_objects": 14225}, code_evidence=["aggregate_scenecad_scan2cad.py"]),
        _source("SceneSmith", f"{IM}/SceneSmith_exported", "json_single.json",
                room_geometry="floor-plan mesh AABB; Room/House frames", room_type="frame-name derived",
                size_semantics="visual glTF/SDF mesh local extent; origin bounds must be retained",
                extent_kind="full", units="meters", position_reference="DMD asset origin; mixed bottom and centered wall assets",
                rotation="AngleAxis -> Rz Ry Rx Euler degrees", up_handedness="DMD Z-up; glTF Y-up -> (x,-z,y)",
                scale="SDF visual scale/node transforms applied during extent extraction", meshes="geometry metadata/DMD; full tar meshes not present",
                blockers="v1 46 wall assets have wrong center assumption; per-asset bounds/front missing",
                declared_counts={"scenes": 210, "rooms": 364, "objects": 19360}, code_evidence=["kits/scenesmith_aggregate.py", "kits/scenesmith_extract_geometry_metadata.py"]),
        _source("SpatialLM", f"{IM}/SpatialLM_exported", "spatiallm.json",
                room_geometry="wall-loop polygon, known invalid rings; source floor plan", room_type="source room category",
                size_semantics="Bbox scale_x/y/z local extents", extent_kind="full", units="meters",
                position_reference="box geometric center", rotation="angle_z radians, yaw-only format",
                up_handedness="Z-up source frame", scale="box extents already final geometry", meshes="not present; layout archive retained",
                blockers="semantic front calibrated in v1 but not per-asset proven; cross-export source-house duplicates require common split",
                declared_counts={"scenes": 12328, "rooms": 54775}, code_evidence=["aggregate_spatiallm.py"]),
        _source("Structured3D", f"{IM}/Structured3D_exported", "structured3d.jsonl",
                room_geometry="annotation junction/line floor polygons, opening planes, house-room split", room_type="source semantic room labels",
                size_semantics="raw_coeffs along raw_basis, not heuristic exported axis reorder", extent_kind="raw half extents",
                units="raw bbox millimeters; exported meters", position_reference="raw bbox centroid; exported room-local center",
                rotation="raw basis rows; geometric vs semantic front not guaranteed", up_handedness="right-handed Z-up",
                scale="raw half extents × 2/1000 once", meshes="not present; annotation and rendered labels retained",
                blockers="heuristic yaw axes/front, 68 zero-sized boxes, source errata, unknown category/tilt",
                declared_counts={"houses": 3488, "rooms": 22894, "assigned_objects": 398485}, code_evidence=["kits/convert_structured3d.py"]),
        _source("SAGE10k", f"{SAGE}/SAGE10k", "json_single.json",
                room_geometry="authored rectangle/wall loop and ceiling height", room_type="source room_type",
                size_semantics="local dimensions width/length/height; raw mesh scale/frame must verify",
                extent_kind="full", units="positions/dimensions meters; placement_constraints XY centimeters",
                position_reference="asset bottom-center origin; raw source physics pose rotates that origin",
                rotation="Rz Ry Rx degrees incl real pitch/roll; README nonsemantic-tilt claim is not sufficient",
                up_handedness="Z-up; right-handed from source matrix/render code", scale="object-generation scale/mesh size requires raw asset check",
                support="place_id floor/wall/instance retained in per-scene source", relations="placement_constraints include exact target pose: exclude from main condition",
                meshes="local scene metadata; mesh catalog/semantic-front completeness unverified",
                blockers="semantic front currently empirical +Y; significant physics tilt must explicitly reject; support surfaces not bbox tops",
                declared_counts={"scenes": 10000, "rooms": 10027, "objects": 853428}, code_evidence=["aggregate_scenes.py", "sage10k/kits/tex_utils_local.py"]),
        _source("HSSD", "XXXpilar__hssd_clean", "objects.csv",
                room_geometry="regions.csv polygon/floor/ceiling; 2351 regions", room_type="human region label",
                size_semantics="rigid mesh-axis OBB half extents; articulated rows lack box", extent_kind="half", units="meters",
                position_reference="obb_center; translation is asset_local pivot; openings metadata center may be wrong",
                rotation="obb_rotation_wxyz; full rigid transform", up_handedness="right-handed Y-up -> (x,-z,y)",
                scale="obb_half_extents already scaled; NEVER multiply non_uniform_scale again",
                support="asset support/affordance metadata, not verified mesh surfaces", relations="unavailable",
                meshes="GLB stages/objects/openings plus URDFs present", blockers="per-asset front/up not retained in catalog; empirical v1 fronts differ; opening/pivot and articulated twins need reliable repair",
                declared_counts={"scenes": 168, "regions": 2351, "objects": 56647}, code_evidence=[]),
        _source("MultiScan", "XXXpilar__multiscan-clean", "objects.csv",
                room_geometry="partial floor-slab convex hull; slab top not a verified contact plane", room_type="source scan room_type",
                size_semantics="annotation OBB axes local extents plus annotated front/up", extent_kind="half", units="meters",
                position_reference="obb_center -> each own full-height bottom center", rotation="obb_axes plus annotated front/up",
                up_handedness="right-handed aligned Z-up", scale="annotated box already metric; no extra scale",
                semantic_front="explicit source front vector; admitted only horizontal OBB-aligned fronts",
                support="geometry-derived rests_on_floor/supported_by; not trusted physical labels",
                relations="geometry-derived; omitted to prevent reference-layout leakage", meshes="273 PLY+textured OBJ scans, extracted per-object assets not built",
                blockers="strict tilt/frame reject; incomplete boundary/floor prevent full room-validity claims",
                eligibility="full_geometry_supervision_after_strict_filter", declared_counts={"scans": 273, "scenes": 117, "objects": 10957},
                code_evidence=[]),
    ]


def _file_evidence(path):
    text = path.read_bytes()
    return {"path": path.name, "bytes": len(text), "sha256": hashlib.sha256(text).hexdigest()}


def _sample(path):
    """Read at most 4 MiB of a large array; never load whole corpus into RAM."""
    if path.suffix == ".csv":
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            first = next(reader, {})
            count = bool(first) + sum(1 for _ in reader)
        return {"measured_records": count, "first_record_keys": sorted(first)}
    with path.open(encoding="utf-8") as fh:
        prefix = fh.read(4 * 1024 * 1024).lstrip()
    prefix = prefix[1:].lstrip() if prefix.startswith("[") else prefix
    first, _ = json.JSONDecoder().raw_decode(prefix)
    rooms = first.get("rooms") or ([first["room"]] if "room" in first else [])
    room = rooms[0] if rooms else first.get("room_context", {})
    objects = room.get("furniture") or first.get("layout", {}).get("floor_layout", {}).get("objects") or []
    return {"first_record_keys": sorted(first), "first_room_keys": sorted(room),
            "first_object_keys": sorted(objects[0]) if objects else [], "sampling": "first record only; max 4 MiB"}


def audit_sources(source_root):
    root = Path(source_root).resolve()
    entries = []
    for definition in source_catalog():
        base = root / definition["directory"]
        evidence = []
        for rel in ["README.md", "manifest.json", "dataset_manifest.json", *definition["code_evidence"]]:
            path = base / rel
            if path.is_file():
                evidence.append({**_file_evidence(path), "path": rel})
        sample = {}
        path = base / definition["sample_file"]
        if path.is_file():
            try:
                sample = _sample(path)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                sample = {"sample_error": str(exc)}
        entries.append({**definition, "present": base.is_dir(), "evidence": evidence, "actual_sample": sample,
                        "count_scope": "declared_counts are retained source documentation, not a new full-corpus recount"})
    return {"schema_version": "fastfill.v2.source_audit", "source_root": str(root),
            "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "top_directories": [{"name": name, "present": (root / name).is_dir()} for name in TOP_DIRECTORIES],
            "sources": entries, "scope": "README+converter+bounded real record inventory; CSV rows counted; no in-place normalization",
            "source_data_modified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/Volumes/harddisk/3D_Room_Collections"))
    parser.add_argument("--output", type=Path, help="New JSON output file outside source data")
    args = parser.parse_args()
    report = audit_sources(args.source_root)
    if args.output:
        path = args.output.resolve()
        protected = (args.source_root.resolve(), CANONICAL_SOURCE_ROOT.resolve())
        if any(path == root or root in path.parents or path in root.parents for root in protected):
            raise ValueError("audit output must be outside read-only source data")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False, allow_nan=False)
            fh.write("\n")
        print(json.dumps({"output": str(path), "sources_present": sum(s["present"] for s in report["sources"])}))
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
