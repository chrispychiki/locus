"""Locus evidence — the store's recorded rrweb into a queryable, replayable local db.

The spine, in the order the data flows:

    store            the operator's bucket, read directly over S3 (inventory lists its slices)
    chunk            one stored chunk -> (visitor, canonical rrweb event) pairs
    load             one incremental store -> db load, chunk manifest and accounting included
    hydrate          raw-first load into events.db (dedup, canonical order)
    slices           grouping by the recorder's slice stamp, slice accounting, rescue
    distill/ (bun)   flat columns, markdown projections, mutation diffs
    sessions         the operator's session definition applied over the flat columns
    render           slice events -> PNG screenshots (headless Chromium)
    replay           the embeddable replay component, slice-set payloads, the default page
"""
