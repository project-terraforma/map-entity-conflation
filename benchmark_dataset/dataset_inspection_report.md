# Boulder/Louisville Reference Dataset Inspection Report

Source archive inspected:

```text
benchmark_dataset/reference_datasets/boulder_louisville_reference.zip
```

Extracted folder:

```text
benchmark_dataset/reference_datasets/extracted/boulder_louisville_sign_and_entrance_gt/
```

The archive title and README describe the dataset as sign and entrance ground-truth positions for Boulder and Louisville. Inspection confirms that it contains field-collected point annotations for signs and doors, plus photos and simple region polygons. It does not contain explicit facade ids, building labels, address labels, street labels, or model confidence labels.

Important geography note: the "Louisville" coordinates in this archive are near longitude `-105.13`, latitude `39.98`, which corresponds to Louisville, Colorado, not Louisville, Kentucky.

## File Inventory

| File | Type | Rows / Count | Geometry Types | Notes |
| --- | --- | ---: | --- | --- |
| `README.txt` | Text | 1 | none | Describes CSV columns as sign and entrance ground-truth positions. |
| `boulder_medium.geojson` | GeoJSON | 1 | Polygon | Boulder region polygon. |
| `louisville_medium.geojson` | GeoJSON | 1 | Polygon | Louisville, CO region polygon; ring is not closed, so GeoPandas rejects it unless read as raw JSON. |
| `boulder_snips_gt_adjusted.csv` | CSV | 349 | point coordinates in columns | Sign/front-door/rear-door annotations. |
| `louisville_snips_gt_adjusted.csv` | CSV | 263 | point coordinates in columns | Sign/window-sign/front-door/rear-door annotations. |
| `Media/` | Images | 485 files | none | Field photos referenced by `image_filename`. |

## README Contents

The included README says:

- two files contain sign and entrance ground-truth positions for Boulder and Louisville
- `POI_name` is the place name
- `overture_id` is the Overture `gers_id`, null if not found in the 2026-01-21 Overture snapshot
- `image_filename` is a photo of the place
- `Type` is `sign`, `front_door`, or `rear_door`
- `Latitude` and `Longitude` are ground-truth positions
- `Exclude` is a Zephr-specific exclusion flag

This supports interpreting the CSVs as manually/field-collected sign and entrance point data. It does not state that facade, building, address, or street matches were manually verified.

## Structured File Inspection

### `boulder_medium.geojson`

- rows: `1`
- geometry type: `Polygon`
- CRS: `EPSG:4326`
- columns:

```text
geometry
```

Sample row:

```text
POLYGON ((-105.28364 40.01606, -105.27038 40.01872, -105.27103 40.02079, -105.28437 40.0181, -105.28364 40.01606))
```

### `louisville_medium.geojson`

- rows: `1`
- geometry type: `Polygon`
- columns:

```text
geometry
```

Sample geometry from raw JSON:

```text
POLYGON-like coordinates:
[-105.13286741692944, 39.980649]
[-105.13286741692944, 39.9759]
[-105.128067, 39.9759]
[-105.128067, 39.980649]
```

Note: the polygon ring is not closed in the file, so GeoPandas/Shapely reports:

```text
Points of LinearRing do not form a closed linestring
```

This is usable as a rough region extent after repair, but should not be treated as curated geometry ground truth.

### `boulder_snips_gt_adjusted.csv`

- rows: `349`
- unique POI names: `135`
- unique non-null Overture ids: `110`
- null `overture_id` rows: `66`
- latitude range: `40.0170823` to `40.0198041`
- longitude range: `-105.2835424` to `-105.2712857`

Columns:

```text
POI_name
overture_id
image_filename
Type
Remarks
Time
Latitude
Longitude
Elevation
Exclude
```

Type counts:

| Type | Count |
| --- | ---: |
| `sign` | 211 |
| `front_door` | 133 |
| `rear_door` | 2 |
| `fornt_door` | 1 |
| `front_)door` | 1 |
| `back_door` | 1 |

Exclude counts:

| Exclude | Count |
| --- | ---: |
| `0` | 349 |

Sample rows:

| POI_name | overture_id | image_filename | Type | Remarks | Latitude | Longitude | Exclude |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| Twig Hair Salon | `322d9aaa-3a17-4deb-b6b9-366942d4698c` | `boulder-signs-03-05-2026_20260326_114301902.jpg` | `sign` | Twig sign | 40.019804 | -105.271286 | 0 |
| Twig Hair Salon | `322d9aaa-3a17-4deb-b6b9-366942d4698c` | `boulder-signs-03-05-2026_20260326_114301902.jpg` | `front_door` | Twig front door | 40.019801 | -105.271299 | 0 |
| Food Lab | `52dd1f13-d71a-4f6c-a5eb-21ceab7ce9c0` | `boulder-signs-03-05-2026_20260326_114354039.jpg` | `sign` | Food Lab sign | 40.019785 | -105.271372 | 0 |
| Food Lab | `52dd1f13-d71a-4f6c-a5eb-21ceab7ce9c0` | `boulder-signs-03-05-2026_20260326_114354039.jpg` | `front_door` | Food Lab front door | 40.019787 | -105.271364 | 0 |
| Boxcar Coffee | `371c7805-b205-4af2-b426-10bc99b633c8` | `boulder-signs-03-05-2026_20260326_114438323.jpg` | `sign` | Boxcar sign | 40.019765 | -105.271467 | 0 |

### `louisville_snips_gt_adjusted.csv`

- rows: `263`
- unique POI names: `93`
- unique non-null Overture ids: `72`
- null `overture_id` rows: `40`
- latitude range: `39.9760387` to `39.9804959`
- longitude range: `-105.1326847` to `-105.128107`

Columns:

```text
POI_name
overture_id
image_filename
Type
Remarks
Time
Latitude
Longitude
Elevation
Exclude
```

Type counts:

| Type | Count |
| --- | ---: |
| `sign` | 124 |
| `front_door` | 78 |
| `window_sign` | 47 |
| `rear_door` | 7 |
| `back_door` | 5 |
| `front door` | 2 |

Exclude counts:

| Exclude | Count |
| --- | ---: |
| `0` | 254 |
| `1` | 9 |

Sample rows:

| POI_name | overture_id | image_filename | Type | Remarks | Latitude | Longitude | Exclude |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| Chad Thielen: Allstate Insurance | `1fe75c4f-7765-4cf6-8ff8-b1da112caf54` | `chad_thielen_PXL_20260128_195537324.jpg` | `sign` | Chad Thielen Sign Post | 39.976256 | -105.132284 | 0 |
| Chad Thielen: Allstate Insurance | `1fe75c4f-7765-4cf6-8ff8-b1da112caf54` | `chad_thielen_PXL_20260128_195537324.jpg` | `window_sign` | Chad Thielen Window | 39.976239 | -105.132310 | 0 |
| Chad Thielen: Allstate Insurance | `1fe75c4f-7765-4cf6-8ff8-b1da112caf54` | `chad_thielen_PXL_20260128_195537324.jpg` | `front_door` | Chad Thielen Front Door | 39.976222 | -105.132308 | 0 |
| Eyeworks | `64d8bc9b-00c8-4130-806c-dfd0a51e7528` | `eyeworks_PXL_20260128_195737673.jpg` | `sign` | Eyeworks banner sign above front door | 39.976370 | -105.132279 | 0 |
| Eyeworks | `64d8bc9b-00c8-4130-806c-dfd0a51e7528` | `eyeworks_PXL_20260128_195737673.jpg` | `front_door` | Eyeworks front door | 39.976383 | -105.132279 | 0 |

## Image File Inspection

The `Media/` folder contains `485` image files:

| Extension | Count |
| --- | ---: |
| `.jpg` | 473 |
| `.jpeg` | 12 |

CSV image references:

| CSV | Unique image references | Found in `Media/` | Missing |
| --- | ---: | ---: | ---: |
| `boulder_snips_gt_adjusted.csv` | 256 | 256 | 0 |
| `louisville_snips_gt_adjusted.csv` | 206 | 205 | 1 |

Missing referenced image:

```text
jasmine-bar-rear-right-door-sign-census-1-28-2026_20260204_115715002.jpg
```

The images provide visual evidence for signs and entrances, but image files alone are not structured facade labels unless a reviewer uses them to create labels.

## Field Availability Checklist

| Field / Label Needed | Exists? | Evidence |
| --- | --- | --- |
| Entrance latitude/longitude | Yes | `Type` values include `front_door`, `rear_door`, `back_door`; coordinates in `Latitude`, `Longitude`. |
| Entrance polygons | No | Only point coordinates are present. |
| Entrance points | Yes | Door rows with latitude/longitude. |
| Storefront points | Partially | Sign/window-sign/front-door points can serve as storefront-related point observations. |
| Facade identifiers | No | No facade id, facade index, building edge id, or selected facade column exists. |
| Facade directions/orientations | No | No bearing, orientation, or normal vector fields exist. |
| Manually verified building matches | No | `overture_id` links to a POI/place, not a verified building id. |
| Verified address labels | No | No address fields are present. |
| Street associations | No | No street id/name association fields are present. |
| Human-reviewed annotations | Partially | `Type`, `Remarks`, `Exclude`, image references, and ground-truth coordinates are annotations. They are not facade/building/address correctness labels. |
| Confidence labels | No | No confidence or model score columns exist. |
| POI-to-Overture id link | Partially | `overture_id` exists, but some rows are null. |
| Exclusion metadata | Yes | `Exclude` exists. |

## Dataset Classification

This dataset is best classified as:

```text
partially labeled reference data
```

It is not only raw Overture data, because it includes field-collected sign and entrance positions, photos, remarks, and exclusion flags.

It is not a complete golden benchmark for Problems 1 or 2, because it does not include manually verified:

- building matches
- address matches
- street associations
- facade ids
- facade orientations
- correct/incorrect labels for pipeline outputs

It does not appear to be synthetic/generated data. The CSVs reference photos and collection times, and the README calls the coordinates ground-truth positions.

## Suitability for Problem 1 Evaluation

Problem 1 evaluates POI to address, building, and street conflation.

This dataset is not sufficient as a true Problem 1 golden benchmark because it lacks:

- verified address labels
- verified building ids
- verified street names or segment ids
- human-reviewed correctness labels for POI-address-building-street relationships

What it can support:

- checking whether a pipeline POI can be linked to a known `overture_id`
- spatial sanity checks between a POI and observed signs/doors
- identifying POIs that have field-observed storefront/entrance evidence

It cannot directly score Problem 1 address/building/street accuracy without additional manual labeling.

## Suitability for Problem 2 Evaluation

Problem 2 evaluates POI to facade/storefront matching.

This dataset is valuable for Problem 2, but it is not a complete facade-level golden dataset.

Useful Problem 2 signals:

- ground-truth sign point coordinates
- ground-truth front/rear/back door point coordinates
- window sign point coordinates in Louisville
- POI names
- Overture POI ids where available
- image evidence for manual review
- exclusion flag for some signs

Missing for direct facade-level evaluation:

- selected facade id
- selected building edge id
- facade orientation
- entrance-to-building association
- reviewer label saying which predicted facade is correct

Therefore, true facade-level evaluation is not directly possible from the fields alone. A direct evaluation would require deriving or manually labeling the facade edge corresponding to each sign/door point.

## Proxy Evaluations That Are Possible

If explicit facade labels are not added, the dataset can still support useful proxy evaluation:

1. **Predicted facade to observed sign/door distance**
   - For each matched `overture_id`, measure distance from the predicted facade edge to ground-truth sign/front-door points.
   - Lower distance is better, but not a proof of correctness.

2. **Derived nearest-edge pseudo-label**
   - For each sign/front-door point, assign the nearest building edge as a pseudo-facade label.
   - Compare the pipeline-selected facade id to this derived edge.
   - This remains heuristic, not verified ground truth.

3. **Entrance/sign support rate**
   - Count how often the predicted facade is within a small threshold of any front-door/sign point for the same POI.

4. **Ambiguity reduction validation**
   - Compare baseline and re-ranked facade outputs by distance to observed front-door/sign points.
   - Useful for ranking experiments, but still not human-validated accuracy.

5. **Manual review packet generation**
   - Use the photos, sign/door points, building footprints, and predicted facades to produce reviewer packets.
   - Human reviewers can then create true `correct`, `incorrect`, or `ambiguous` labels.

## Recommendation

Use this dataset as a strong seed for a real benchmark, not as finalized golden labels.

Recommended next step:

- Convert sign/front-door rows into geospatial point layers.
- Join rows to Overture places by `overture_id`.
- Generate candidate facades from building footprints.
- Create a reviewer workflow where humans verify the correct facade edge for each POI.
- Store the verified result as explicit facade labels, for example:

```text
overture_id
verified_building_id
verified_facade_id
verified_label
reviewer_notes
source_image_filename
```

Until those labels exist, Problem 2 evaluation should be described as proxy evaluation using field-observed signs/entrances, not as true facade accuracy.
