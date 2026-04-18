# Grounding Data Pipeline: Privacy Notice

## What is collected

### Trajectory logging (`QONTINUI_TRAJECTORY_LOGGER=on`)

When trajectory logging is enabled, the following data is captured for every
GUI action executed during a workflow:

- **Full-resolution screenshots** of the entire primary monitor, taken
  immediately before and after each action.
- **Action metadata**: action type (click, type, scroll, etc.), target
  coordinates, typed text, timestamps, and success/failure status.
- **Element bounding boxes** detected by OmniParser (if enabled via
  `QONTINUI_OMNIPARSER_ENABLED=true`), including interactability labels.
- **World State Verifier verdicts** (if the WSM service is reachable),
  including confidence scores and observations.

All data is written to `grounding.jsonl` and the `images/` directory inside
the configured `QONTINUI_EXPORT_DIR`.

### Static capture pipeline (`capture-grounding-data.ts`)

When the Playwright capture script is run manually:

- **Screenshots** of the qontinui-web application are captured at multiple
  viewport sizes.
- **Element bounding boxes** extracted via DOM introspection
  (`getBoundingClientRect`) are stored alongside each screenshot.

This captures only the qontinui-web UI itself — no user data is involved.

### Existing training data export (`QONTINUI_EXPORT_TRAINING_DATA=true`)

The pre-existing training export system captures screenshots and action
records during automation runs. See `TRAINING_DATA_SYSTEM.md` for details.

## What is NOT done

- Screenshots are **not scrubbed**, redacted, or anonymised.
- **No personal data pseudonymisation** is applied.
- **No data is transmitted** to external servers unless the user explicitly
  configures a remote export target.
- No attempt is made to detect or mask sensitive content (passwords, PII,
  financial data, etc.) visible on screen.

Pseudonymisation is intentionally omitted — it provides false comfort and is
out of scope for this data pipeline.

## Opt-in only

All data collection mechanisms are strictly opt-in:

| Feature | Activation |
|---|---|
| Trajectory logging | `QONTINUI_TRAJECTORY_LOGGER=on` |
| WSM success labelling (inside trajectory logging) | `QONTINUI_WSM_ENABLED=1` (default when trajectory logging is on) |
| Static capture | Manual run of `capture-grounding-data.ts` |
| Training data export | `QONTINUI_EXPORT_TRAINING_DATA=true` |

`QONTINUI_WSM_ENABLED` can be set to `0`/`false`/`off`/`no` to skip the
World State Verifier step and force the pixel-diff heuristic. This does
not disable trajectory logging itself — only the source of the
`success_source` label on each record.

**No data collection occurs by default.**

## Data storage

All collected data is stored locally in the directory specified by
`QONTINUI_EXPORT_DIR` (or the run directory if not set). Users are
responsible for:

- Managing filesystem permissions on the export directory.
- Reviewing captured screenshots before sharing or uploading datasets.
- Deleting data when it is no longer needed.

## Disk usage

Trajectory logging writes one PNG screenshot per action. At typical
1920x1080 resolution, each PNG is approximately 1-3 MB. The
`QONTINUI_TRAJECTORY_MAX_RECORDS` environment variable (default: 500)
caps the number of records per session to prevent disk fill.

The `grounding.jsonl` file is rotated at 100 MB.

## Recommendations

- **Do not enable trajectory logging** when working with sensitive data
  visible on screen (credentials, personal information, financial data).
- **Review captured screenshots** before sharing datasets with others.
- **Use filesystem permissions** to restrict access to the export directory.
- **Delete datasets** when they are no longer needed for training.
