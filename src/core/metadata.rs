// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use crate::core::manifest::{PartitionSpec, Schema, SortOrder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "kebab-case")]
pub struct SnapshotLogEntry {
    pub timestamp_ms: i64,
    pub snapshot_id: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "kebab-case")]
pub struct MetadataLogEntry {
    pub timestamp_ms: i64,
    pub metadata_file: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "kebab-case")]
pub struct Snapshot {
    pub snapshot_id: i64,
    /// Iceberg's parser rejects an explicit JSON `null` for this key, so it is
    /// omitted entirely for a root snapshot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_snapshot_id: Option<i64>,
    pub timestamp_ms: i64,
    pub sequence_number: Option<i64>,
    pub summary: HashMap<String, String>,
    pub manifest_list: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema_id: Option<i32>,

    // V3 Row Lineage (Iceberg spec v3 required)
    /// The first _row_id assigned to the first row in the first data file
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_row_id: Option<i64>,

    /// The upper bound of the number of rows with assigned row IDs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub added_rows: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "kebab-case")]
pub struct TableMetadata {
    pub format_version: i32,
    pub table_uuid: String,
    pub location: String,
    pub last_sequence_number: i64,
    pub last_updated_ms: i64,
    pub last_column_id: i32,

    /// Highest partition field id ever assigned. Iceberg requires this key and
    /// reserves ids from 1000 upwards, so an unpartitioned table reports 999.
    #[serde(default = "default_last_partition_id")]
    pub last_partition_id: i32,

    // Schema
    pub current_schema_id: i32,
    pub schemas: Vec<Schema>,

    // Partitioning
    pub default_spec_id: i32,
    pub partition_specs: Vec<PartitionSpec>,

    // Sorting
    pub default_sort_order_id: i32,
    pub sort_orders: Vec<SortOrder>,

    // Primary Key (Iceberg Identifier Fields)
    #[serde(
        rename = "identifier-field-ids",
        default,
        skip_serializing_if = "Vec::is_empty"
    )]
    pub identifier_field_ids: Vec<i32>,

    // Properties
    #[serde(default)]
    pub properties: HashMap<String, String>,

    // Snapshots
    #[serde(default)]
    pub current_snapshot_id: Option<i64>,
    #[serde(default)]
    pub snapshots: Vec<Snapshot>,

    #[serde(default)]
    pub snapshot_log: Vec<SnapshotLogEntry>,

    #[serde(default)]
    pub metadata_log: Vec<MetadataLogEntry>,

    // V3 Row Lineage (Iceberg spec v3 required)
    /// A long higher than all assigned row IDs; the next snapshot's first-row-id
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_row_id: Option<i64>,
}

/// First partition field id Iceberg assigns; unpartitioned tables report one less.
const ICEBERG_PARTITION_DATA_ID_START: i32 = 1000;

fn default_last_partition_id() -> i32 {
    ICEBERG_PARTITION_DATA_ID_START - 1
}

impl TableMetadata {
    /// Create a minimal metadata skeleton (used by catalogs that only know the location)
    pub fn minimal(location: String) -> Self {
        Self {
            format_version: 2,
            location,
            last_partition_id: default_last_partition_id(),
            ..Default::default()
        }
    }

    pub fn new(
        format_version: i32,
        uuid: String,
        location: String,
        schema: Schema,
        partition_spec: PartitionSpec,
        sort_order: SortOrder,
    ) -> Self {
        Self {
            format_version,
            table_uuid: uuid,
            location,
            last_sequence_number: 0,
            last_updated_ms: chrono::Utc::now().timestamp_millis(),
            last_column_id: schema.fields.iter().map(|f| f.id).max().unwrap_or(0),
            last_partition_id: partition_spec
                .fields
                .iter()
                .filter_map(|f| f.field_id)
                .max()
                .unwrap_or(default_last_partition_id())
                .max(default_last_partition_id()),
            current_schema_id: schema.schema_id,
            schemas: vec![schema.clone()],
            default_spec_id: partition_spec.spec_id,
            partition_specs: vec![partition_spec],
            default_sort_order_id: sort_order.order_id,
            sort_orders: vec![sort_order],
            properties: HashMap::new(),
            current_snapshot_id: None,
            snapshots: Vec::new(),
            snapshot_log: Vec::new(),
            metadata_log: Vec::new(),
            next_row_id: if format_version >= 3 { Some(0) } else { None },
            identifier_field_ids: schema.identifier_field_ids.clone(),
        }
    }

    /// Add a new snapshot and update current_snapshot_id
    pub fn add_snapshot(&mut self, snapshot: Snapshot) {
        self.last_updated_ms = snapshot.timestamp_ms;
        self.current_snapshot_id = Some(snapshot.snapshot_id);
        // Iceberg validates that `last-sequence-number` covers every snapshot.
        if let Some(sequence_number) = snapshot.sequence_number {
            self.last_sequence_number = self.last_sequence_number.max(sequence_number);
        }
        self.snapshot_log.push(SnapshotLogEntry {
            timestamp_ms: snapshot.timestamp_ms,
            snapshot_id: snapshot.snapshot_id,
        });
        self.snapshots.push(snapshot);
    }

    /// Merge a snapshot monotonically and idempotently into `self`.
    ///
    /// - If a snapshot with `snapshot.snapshot_id` is already present, avoids adding
    ///   a duplicate entry (still advancing `next_row_id` if higher).
    /// - If `snapshot.sequence_number` trails an already-published snapshot (delayed
    ///   concurrent metadata writer), preserves `current_snapshot_id`,
    ///   `last_updated_ms`, and `last_sequence_number` at the newer head while
    ///   retaining the historical snapshot entry.
    /// - Returns `true` if `self` changed.
    pub fn merge_snapshot_monotonic(
        &mut self,
        mut snapshot: Snapshot,
        candidate_next_row_id: Option<i64>,
    ) -> bool {
        if self
            .snapshots
            .iter()
            .any(|s| s.snapshot_id == snapshot.snapshot_id)
        {
            if self.format_version >= 3 {
                if let Some(n) = candidate_next_row_id {
                    let prev = self.next_row_id.unwrap_or(0);
                    if n > prev {
                        self.next_row_id = Some(n);
                        return true;
                    }
                }
            }
            return false;
        }

        let head_seq = self
            .snapshots
            .iter()
            .filter_map(|s| s.sequence_number)
            .max()
            .unwrap_or(self.last_sequence_number);
        let snap_seq = snapshot.sequence_number.unwrap_or(0);

        if snap_seq >= head_seq {
            snapshot.parent_snapshot_id = self.current_snapshot_id;
            self.current_snapshot_id = Some(snapshot.snapshot_id);
            self.last_updated_ms = self.last_updated_ms.max(snapshot.timestamp_ms);
            self.last_sequence_number = self.last_sequence_number.max(snap_seq);
        } else {
            self.last_sequence_number = self.last_sequence_number.max(snap_seq);
        }

        self.snapshot_log.push(SnapshotLogEntry {
            timestamp_ms: snapshot.timestamp_ms,
            snapshot_id: snapshot.snapshot_id,
        });
        self.snapshots.push(snapshot);

        if self.format_version >= 3 {
            if let Some(n) = candidate_next_row_id {
                self.next_row_id = Some(self.next_row_id.unwrap_or(0).max(n));
            }
        }
        true
    }

    /// Add a new schema
    pub fn add_schema(&mut self, schema: Schema) {
        self.last_column_id = schema
            .fields
            .iter()
            .map(|f| f.id)
            .max()
            .unwrap_or(self.last_column_id);
        self.current_schema_id = schema.schema_id;
        self.schemas.push(schema);
    }

    /// Add a new partition spec
    pub fn add_partition_spec(&mut self, spec: PartitionSpec) {
        self.default_spec_id = spec.spec_id;
        if let Some(field_id) = spec.fields.iter().filter_map(|f| f.field_id).max() {
            self.last_partition_id = self.last_partition_id.max(field_id);
        }
        self.partition_specs.push(spec);
    }

    /// Fill in the invariants Iceberg's `TableMetadataParser` enforces but that
    /// older BenoStreamDB manifests may not have recorded.
    ///
    /// Kept at the write boundary (rather than in the constructors) so tables
    /// written by earlier releases are upgraded as they are next committed.
    fn normalize_iceberg_metadata(&mut self) {
        // Iceberg rejects an empty `sort-orders` array.
        if self.sort_orders.is_empty() {
            self.sort_orders.push(SortOrder {
                order_id: self.default_sort_order_id,
                fields: Vec::new(),
            });
        }

        // The default partition spec must exist, even when unpartitioned.
        if self.partition_specs.is_empty() {
            self.partition_specs.push(PartitionSpec {
                spec_id: self.default_spec_id,
                fields: Vec::new(),
            });
        }

        // Every partition field needs a concrete `source-id` and `field-id`:
        // Iceberg parses both as required integers.
        let mut next_field_id = self.last_partition_id.max(default_last_partition_id()) + 1;
        let mut max_field_id = self.last_partition_id.max(default_last_partition_id());
        for spec in &mut self.partition_specs {
            for field in &mut spec.fields {
                if field.source_id.is_none() {
                    field.source_id = field.source_ids.first().copied();
                }
                if field.field_id.is_none() {
                    field.field_id = Some(next_field_id);
                    next_field_id += 1;
                }
                if let Some(field_id) = field.field_id {
                    max_field_id = max_field_id.max(field_id);
                }
            }
        }
        self.last_partition_id = max_field_id;

        // `last-sequence-number` must cover every snapshot in the table.
        let max_sequence_number = self
            .snapshots
            .iter()
            .filter_map(|snapshot| snapshot.sequence_number)
            .max()
            .unwrap_or(0);
        self.last_sequence_number = self.last_sequence_number.max(max_sequence_number);

        // Row lineage (`first-row-id` / `added-rows`) is a v3 snapshot field.
        // Emitting it in a v2 table is outside the spec, even though tolerant
        // readers ignore it.
        if self.format_version < 3 {
            for snapshot in &mut self.snapshots {
                snapshot.first_row_id = None;
                snapshot.added_rows = None;
            }
        } else {
            // Every v3 snapshot must carry a row range. A snapshot that adds no
            // rows — a delete, a compaction, a schema change — records
            // `first-row-id` = the next unassigned row id with `added-rows` = 0.
            // A missing range is not inferable by a reader (it cannot tell which
            // `_row_id` the snapshot's rows start at), so it is backfilled here
            // rather than emitted as a lineage hole. The oracle in
            // `tests/iceberg_rust_oracle` is what caught this.
            let mut cursor = 0i64;
            for snapshot in &mut self.snapshots {
                match (snapshot.first_row_id, snapshot.added_rows) {
                    (Some(first), Some(added)) => {
                        cursor = cursor.max(first.saturating_add(added));
                    }
                    (Some(first), None) => {
                        snapshot.added_rows = Some(0);
                        cursor = cursor.max(first);
                    }
                    _ => {
                        snapshot.first_row_id = Some(cursor);
                        snapshot.added_rows = Some(0);
                    }
                }
            }
        }
    }
}

/// Outcome returned by a closure passed to [`TableMetadata::commit_update_atomic`].
#[derive(Debug, Clone)]
pub enum MetadataUpdateOutcome {
    /// Persist the updated [`TableMetadata`] to a new `v{N}.metadata.json` file.
    Updated(TableMetadata),
    /// The latest metadata already contains this commit (idempotent tail resume).
    AlreadyCurrent(TableMetadata),
}

impl TableMetadata {
    /// Discover the highest `metadata/v{N}.metadata.json` version number in `storage`.
    pub async fn latest_version_in_store(
        storage: &dyn object_store::ObjectStore,
    ) -> anyhow::Result<Option<i32>> {
        use futures::StreamExt;

        let prefix = object_store::path::Path::from("metadata");
        let mut stream = storage.list(Some(&prefix));
        let mut best: Option<i32> = None;
        while let Some(item) = stream.next().await {
            let meta = match item {
                Ok(m) => m,
                Err(object_store::Error::NotFound { .. }) => continue,
                Err(e) => return Err(e.into()),
            };
            if let Some(name) = meta.location.filename() {
                if let Some(v_num) = name
                    .strip_prefix('v')
                    .and_then(|s| s.strip_suffix(".metadata.json"))
                    .and_then(|s| s.parse::<i32>().ok())
                {
                    best = Some(best.map_or(v_num, |b| b.max(v_num)));
                }
            }
        }
        Ok(best)
    }

    /// Monotonically advance `metadata/version-hint.text` using conditional CAS when
    /// supported by the backing store so delayed concurrent writers cannot regress it.
    pub async fn advance_version_hint(
        storage: &dyn object_store::ObjectStore,
        version: i32,
    ) -> anyhow::Result<()> {
        let hint_path = object_store::path::Path::from("metadata/version-hint.text");
        for attempt in 0..4usize {
            let existing = storage.get(&hint_path).await;
            let payload: object_store::PutPayload = version.to_string().into();

            let put_res = match existing {
                Ok(res) => {
                    let e_tag = res.meta.e_tag.clone();
                    let obj_version = res.meta.version.clone();
                    let current_version = res
                        .bytes()
                        .await
                        .ok()
                        .and_then(|bytes| String::from_utf8(bytes.to_vec()).ok())
                        .and_then(|value| value.trim().parse::<i32>().ok());
                    if current_version.is_some_and(|current| current >= version) {
                        return Ok(());
                    }
                    if e_tag.is_some() || obj_version.is_some() {
                        let mode = object_store::PutMode::Update(object_store::UpdateVersion {
                            e_tag,
                            version: obj_version,
                        });
                        storage.put_opts(&hint_path, payload, mode.into()).await
                    } else {
                        storage.put(&hint_path, payload).await
                    }
                }
                Err(object_store::Error::NotFound { .. }) => {
                    storage
                        .put_opts(&hint_path, payload, object_store::PutMode::Create.into())
                        .await
                }
                Err(e) => {
                    if attempt + 1 < 4 {
                        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
                        continue;
                    }
                    return Err(e.into());
                }
            };

            match put_res {
                Ok(_) => return Ok(()),
                Err(object_store::Error::NotImplemented) => {
                    storage.put(&hint_path, version.to_string().into()).await?;
                    return Ok(());
                }
                Err(
                    object_store::Error::AlreadyExists { .. }
                    | object_store::Error::Precondition { .. },
                ) => {
                    if attempt + 1 < 4 {
                        continue;
                    }
                    return Ok(());
                }
                Err(e) => {
                    if attempt + 1 < 4 {
                        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
                        continue;
                    }
                    return Err(e.into());
                }
            }
        }
        Ok(())
    }

    /// Save metadata to ObjectStore as a new version `v<X>.metadata.json` using
    /// `PutMode::Create` when supported by the backing store.
    pub async fn save_to_store(
        &mut self,
        storage: &dyn object_store::ObjectStore,
        version: i32,
    ) -> anyhow::Result<String> {
        self.normalize_iceberg_metadata();
        let path_str = format!("metadata/v{}.metadata.json", version);
        let path = object_store::path::Path::from(path_str.as_str());
        let json = serde_json::to_vec(self)?;

        match storage
            .put_opts(
                &path,
                json.clone().into(),
                object_store::PutMode::Create.into(),
            )
            .await
        {
            Ok(_) => {}
            Err(object_store::Error::NotImplemented) => {
                storage.put(&path, json.into()).await?;
            }
            Err(e) => return Err(e.into()),
        }

        Self::advance_version_hint(storage, version).await?;
        Ok(path_str)
    }

    /// Atomically update `TableMetadata` in `storage` using CAS (`PutMode::Create`)
    /// with bounded conflict-retry discovery of the highest `v{N}.metadata.json`.
    pub async fn commit_update_atomic<F>(
        storage: &dyn object_store::ObjectStore,
        preferred_version: i32,
        mut build_fn: F,
    ) -> anyhow::Result<(String, Self)>
    where
        F: FnMut(Option<Self>) -> anyhow::Result<MetadataUpdateOutcome>,
    {
        for attempt in 0..64usize {
            let latest_ver = Self::latest_version_in_store(storage).await?;
            let current_meta = match latest_ver {
                Some(v) => {
                    let p =
                        object_store::path::Path::from(format!("metadata/v{}.metadata.json", v));
                    Self::load_from_store(storage, &p).await.ok()
                }
                None => Self::load_latest(storage).await.ok(),
            };

            match build_fn(current_meta)? {
                MetadataUpdateOutcome::AlreadyCurrent(meta) => {
                    let ver = latest_ver.unwrap_or(preferred_version.max(1));
                    let _ = Self::advance_version_hint(storage, ver).await;
                    return Ok((format!("metadata/v{}.metadata.json", ver), meta));
                }
                MetadataUpdateOutcome::Updated(mut next_meta) => {
                    next_meta.normalize_iceberg_metadata();
                    let next_ver = latest_ver
                        .map(|v| (v + 1).max(preferred_version))
                        .unwrap_or(preferred_version)
                        .max(1);
                    let path_str = format!("metadata/v{}.metadata.json", next_ver);
                    let path = object_store::path::Path::from(path_str.as_str());
                    let json = serde_json::to_vec(&next_meta)?;

                    match storage
                        .put_opts(
                            &path,
                            json.clone().into(),
                            object_store::PutMode::Create.into(),
                        )
                        .await
                    {
                        Ok(_) => {
                            Self::advance_version_hint(storage, next_ver).await?;
                            return Ok((path_str, next_meta));
                        }
                        Err(object_store::Error::NotImplemented) => {
                            storage.put(&path, json.into()).await?;
                            Self::advance_version_hint(storage, next_ver).await?;
                            return Ok((path_str, next_meta));
                        }
                        Err(
                            object_store::Error::AlreadyExists { .. }
                            | object_store::Error::Precondition { .. },
                        ) => {
                            let backoff_ms = ((attempt as u64) % 8) + 1;
                            tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                            continue;
                        }
                        Err(e) => return Err(e.into()),
                    }
                }
            }
        }

        anyhow::bail!("Exceeded max CAS attempts (64) committing TableMetadata update")
    }

    /// Load metadata from ObjectStore
    pub async fn load_from_store(
        storage: &dyn object_store::ObjectStore,
        path: &object_store::path::Path,
    ) -> anyhow::Result<Self> {
        let res = storage.get(path).await?;
        let bytes = res.bytes().await?;
        let metadata: Self = serde_json::from_slice(&bytes)?;
        Ok(metadata)
    }

    /// Load the latest metadata version using `version-hint.text`, verifying against
    /// newer `v{N}.metadata.json` files if the hint lags or is absent.
    pub async fn load_latest(storage: &dyn object_store::ObjectStore) -> anyhow::Result<Self> {
        let hint_path = object_store::path::Path::from("metadata/version-hint.text");
        if let Ok(res) = storage.get(&hint_path).await {
            if let Ok(bytes) = res.bytes().await {
                if let Ok(version_str) = String::from_utf8(bytes.to_vec()) {
                    if let Ok(version) = version_str.trim().parse::<i32>() {
                        let next_path = object_store::path::Path::from(format!(
                            "metadata/v{}.metadata.json",
                            version + 1
                        ));
                        if storage.head(&next_path).await.is_ok() {
                            if let Ok(Some(max_v)) = Self::latest_version_in_store(storage).await {
                                let path = object_store::path::Path::from(format!(
                                    "metadata/v{}.metadata.json",
                                    max_v
                                ));
                                if let Ok(meta) = Self::load_from_store(storage, &path).await {
                                    return Ok(meta);
                                }
                            }
                        }
                        let path = object_store::path::Path::from(format!(
                            "metadata/v{}.metadata.json",
                            version
                        ));
                        if let Ok(meta) = Self::load_from_store(storage, &path).await {
                            return Ok(meta);
                        }
                    }
                }
            }
        }

        if let Some(max_v) = Self::latest_version_in_store(storage).await? {
            let path = object_store::path::Path::from(format!("metadata/v{}.metadata.json", max_v));
            return Self::load_from_store(storage, &path).await;
        }

        let res = storage.get(&hint_path).await?;
        let bytes = res.bytes().await?;
        let version_str = String::from_utf8(bytes.to_vec())?;
        let version = version_str.trim().parse::<i32>()?;
        let path = object_store::path::Path::from(format!("metadata/v{}.metadata.json", version));
        Self::load_from_store(storage, &path).await
    }
}
