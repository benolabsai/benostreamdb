// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Result;
use crate::core::manifest::{ManifestManager, PartitionSpec};
use super::Table;

impl Table {
    /// Explicit Schema Evolution: Add a new column
    pub async fn add_column(&self, name: &str, data_type: arrow::datatypes::DataType) -> Result<()> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (mut manifest, _, _) = manifest_manager.load_latest_full().await?;
        
        let mut current_schema = if let Some(schema) = manifest.schemas.iter().find(|s| s.schema_id == manifest.current_schema_id) {
             schema.clone()
        } else {
             // Bootstrap schema from Arrow schema if not tracking yet
             let arrow_schema = self.arrow_schema();
             crate::core::manifest::Schema::new(0, arrow_schema.fields().iter().enumerate().map(|(i, f)| {
                 crate::core::manifest::SchemaField {
                     id: i as i32 + 1,
                     name: f.name().clone(),
                     type_str: f.data_type().to_string(),
                      required: !f.is_nullable(),
                      fields: Vec::new(),
                      initial_default: None,
                      write_default: None,
                  }
             }).collect(), Vec::new())
        };

        // Check if exists
        if current_schema.fields.iter().any(|f| f.name == name) {
            return Err(anyhow::anyhow!("Column '{}' already exists", name));
        }

        // New Field ID
        let new_id = manifest.last_column_id + 1;
        let new_field = crate::core::manifest::SchemaField::from_arrow_field(
            &arrow::datatypes::Field::new(name, data_type.clone(), true),
            new_id
        );
        current_schema.fields.push(new_field);

        // Update Manifest
        let new_schema_id = manifest.current_schema_id + 1;
        current_schema.schema_id = new_schema_id;
        
        manifest.schemas.push(current_schema.clone());
        manifest.current_schema_id = new_schema_id;
        manifest.last_column_id = new_id;
        
        // Commit Metadata Only Change
        manifest_manager.update_schema(manifest.schemas, manifest.current_schema_id, Some(new_id)).await?;
        println!("Schema Evolution: Added column '{}' (Schema ID: {})", name, new_schema_id);
        
        let new_arrow_schema = current_schema.to_arrow();
        let mut lock = self.schema.write().unwrap();
        *lock = std::sync::Arc::new(new_arrow_schema);
        
        Ok(())
    }

    /// Partition Spec Evolution: Set a new partition spec (Iceberg V2 spec compliance)
    /// 
    /// The previous spec is retained in history for reading old data files.
    /// 
    /// # Arguments
    /// * `fields` - Partition field definitions (source_id, name, transform)
    /// 
    /// # Example
    /// ```no_run
    /// # use hyperstreamdb::core::table::Table;
    /// # use hyperstreamdb::core::manifest::PartitionField;
    /// # async fn example(table: &Table) -> anyhow::Result<()> {
    /// table.update_spec(&[
    ///     PartitionField::new_single(1, Some(1000), "month".into(), "month".into()),
    /// ]).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn update_spec(&self, fields: &[crate::core::manifest::PartitionField]) -> Result<()> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (manifest, _, _) = manifest_manager.load_latest_full().await?;
        
        // Determine next spec_id
        let new_spec_id = manifest.partition_specs.iter()
            .map(|s| s.spec_id)
            .max()
            .unwrap_or(-1) + 1;
        
        // Create new spec
        let new_spec = PartitionSpec {
            spec_id: new_spec_id,
            fields: fields.to_vec(),
        };
        
        // Build updated specs list (append new spec to history)
        let mut updated_specs = manifest.partition_specs.clone();
        updated_specs.push(new_spec);
        
        // Commit the partition spec evolution
        let metadata = crate::core::manifest::CommitMetadata {
            updated_schemas: Some(manifest.schemas.clone()),
            updated_schema_id: Some(manifest.current_schema_id),
            updated_partition_specs: Some(updated_specs),
            updated_default_spec_id: Some(new_spec_id),
            updated_properties: None,
            removed_properties: None,
            updated_sort_orders: Some(manifest.sort_orders.clone()),
            updated_default_sort_order_id: Some(manifest.default_sort_order_id),
            updated_last_column_id: None,
            is_fast_append: false,
        };
        
        manifest_manager.commit(&[], &[], metadata).await?;
        println!("Partition Evolution: New spec ID {} with {} fields", new_spec_id, fields.len());
        
        Ok(())
    }

    /// Explicit Schema Evolution: Drop a column
    pub async fn drop_column(&self, name: &str) -> Result<()> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (mut manifest, _, _) = manifest_manager.load_latest_full().await?;
        
        let mut current_schema = manifest.schemas.iter().find(|s| s.schema_id == manifest.current_schema_id)
            .ok_or_else(|| anyhow::anyhow!("No active schema found for evolution"))?.clone();
            
        if !current_schema.fields.iter().any(|f| f.name == name) {
            return Err(anyhow::anyhow!("Column '{}' does not exist", name));
        }
        
        // Remove field
        current_schema.fields.retain(|f| f.name != name);
        
        // Update Manifest
        let new_schema_id = manifest.current_schema_id + 1;
        current_schema.schema_id = new_schema_id;
        manifest.schemas.push(current_schema.clone());
        manifest.current_schema_id = new_schema_id;
        manifest_manager.update_schema(manifest.schemas, manifest.current_schema_id, Some(manifest.last_column_id)).await?;
        println!("Schema Evolution: Dropped column '{}' (Schema ID: {})", name, new_schema_id);
        
        let new_arrow_schema = current_schema.to_arrow();
        let mut lock = self.schema.write().unwrap();
        *lock = std::sync::Arc::new(new_arrow_schema);
        
        Ok(())
    }
    
    /// Explicit Schema Evolution: Rename a column
    pub async fn rename_column(&self, old_name: &str, new_name: &str) -> Result<()> {
         let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (mut manifest, _, _) = manifest_manager.load_latest_full().await?;

        let mut current_schema = manifest.schemas.iter().find(|s| s.schema_id == manifest.current_schema_id)
            .ok_or_else(|| anyhow::anyhow!("No active schema found for evolution"))?.clone();
            
        if let Some(field) = current_schema.fields.iter_mut().find(|f| f.name == old_name) {
             field.name = new_name.to_string();
        } else {
             return Err(anyhow::anyhow!("Column '{}' does not exist", old_name));
        }

        let new_schema_id = manifest.current_schema_id + 1;
        current_schema.schema_id = new_schema_id;
        manifest.schemas.push(current_schema.clone());
        manifest.current_schema_id = new_schema_id;
        manifest_manager.update_schema(manifest.schemas, manifest.current_schema_id, Some(manifest.last_column_id)).await?;
        println!("Schema Evolution: Renamed '{}' -> '{}' (Schema ID: {})", old_name, new_name, new_schema_id);
        
        let new_arrow_schema = current_schema.to_arrow();
        let mut lock = self.schema.write().unwrap();
        *lock = std::sync::Arc::new(new_arrow_schema);
        
        Ok(())
    }

    /// Explicit Schema Evolution: Update column type (Type Promotion)
    /// Widens the type of an existing column (e.g., int -> long, float -> double)
    pub async fn update_column_type(&self, name: &str, new_type: &str) -> Result<()> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (mut manifest, _, _) = manifest_manager.load_latest_full().await?;
        
        let mut current_schema = manifest.schemas.iter().find(|s| s.schema_id == manifest.current_schema_id)
            .ok_or_else(|| anyhow::anyhow!("No active schema found for evolution"))?.clone();
            
        if let Some(field) = current_schema.fields.iter_mut().find(|f| f.name == name) {
             if !Self::can_promote(&field.type_str, new_type) {
                 return Err(anyhow::anyhow!("Invalid type promotion: {} -> {}", field.type_str, new_type));
             }
             field.type_str = new_type.to_string();
        } else {
             return Err(anyhow::anyhow!("Column '{}' does not exist", name));
        }

        let new_schema_id = manifest.current_schema_id + 1;
        current_schema.schema_id = new_schema_id;
        manifest.schemas.push(current_schema.clone());
        manifest.current_schema_id = new_schema_id;
        
        manifest_manager.update_schema(manifest.schemas, manifest.current_schema_id, Some(manifest.last_column_id)).await?;
        println!("Schema Evolution: Updated column type '{}' to '{}' (Schema ID: {})", name, new_type, new_schema_id);
        
        let new_arrow_schema = current_schema.to_arrow();
        let mut lock = self.schema.write().unwrap();
        *lock = std::sync::Arc::new(new_arrow_schema);
        
        Ok(())
    }

    fn can_promote(old_type: &str, new_type: &str) -> bool {
        match (old_type.to_lowercase().as_str(), new_type.to_lowercase().as_str()) {
            ("int" | "int32", "long" | "int64") => true,
            ("float" | "float32", "double" | "float64") => true,
            (o, n) if (o.contains("decimal") || o.contains("decimal")) && (n.contains("decimal") || n.contains("decimal")) => {
                // Simplified: allow any decimal to any decimal for now (usually widening precision)
                true
            },
            _ => false
        }
    }

    /// Explicit Schema Evolution: Move a column to a new position (0-based index)
    pub async fn move_column(&self, name: &str, new_index: usize) -> Result<()> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (mut manifest, _, _) = manifest_manager.load_latest_full().await?;
        
        let mut current_schema = manifest.schemas.iter().find(|s| s.schema_id == manifest.current_schema_id)
            .ok_or_else(|| anyhow::anyhow!("No active schema found for evolution"))?.clone();
            
        let old_index = current_schema.fields.iter().position(|f| f.name == name)
            .ok_or_else(|| anyhow::anyhow!("Column '{}' does not exist", name))?;
            
        if new_index >= current_schema.fields.len() {
             return Err(anyhow::anyhow!("Invalid new index {}", new_index));
        }
        
        if old_index == new_index {
            return Ok(()); // No-op
        }

        let field = current_schema.fields.remove(old_index);
        current_schema.fields.insert(new_index, field);

        let new_schema_id = manifest.current_schema_id + 1;
        current_schema.schema_id = new_schema_id;
        manifest.schemas.push(current_schema.clone());
        manifest.current_schema_id = new_schema_id;
        
        manifest_manager.update_schema(manifest.schemas, manifest.current_schema_id, Some(manifest.last_column_id)).await?;
        println!("Schema Evolution: Moved column '{}' to index {} (Schema ID: {})", name, new_index, new_schema_id);
        
        let new_arrow_schema = current_schema.to_arrow();
        let mut lock = self.schema.write().unwrap();
        *lock = std::sync::Arc::new(new_arrow_schema);
        
        Ok(())
    }
}
