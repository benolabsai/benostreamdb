package com.hyperstreamdb.spark

import org.apache.spark.sql.connector.write.{DeltaWriter, DeltaWriterFactory}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.write.WriterCommitMessage

// We will use Jackson (available in Spark) to serialize JSON
import com.fasterxml.jackson.databind.ObjectMapper

class HyperStreamPositionDeltaWriterFactory(tableName: String, gpuDevice: String) extends DeltaWriterFactory {
  override def createWriter(partitionId: Int, taskId: Long): DeltaWriter[InternalRow] = {
    new HyperStreamDeltaWriter(tableName, gpuDevice)
  }
}

class HyperStreamDeltaCommitMessage(val status: String) extends WriterCommitMessage

class HyperStreamDeltaWriter(tableName: String, gpuDevice: String) extends DeltaWriter[InternalRow] {
  
  // Accumulate deletes: Map[FilePath, ArrayBuffer[RowPosition]]
  private val deletes = scala.collection.mutable.Map.empty[String, scala.collection.mutable.ArrayBuffer[Long]]

  override def delete(metadata: InternalRow, id: InternalRow): Unit = {
    // Spark passes _file as UTF8String at index 0, _pos as Long at index 1
    val file = metadata.getUTF8String(0).toString
    val pos = metadata.getLong(1)
    deletes.getOrElseUpdate(file, scala.collection.mutable.ArrayBuffer.empty[Long]) += pos
  }

  override def update(metadata: InternalRow, id: InternalRow, row: InternalRow): Unit = {
    // For MERGE/UPDATE, the old record is deleted via position
    val file = metadata.getUTF8String(0).toString
    val pos = metadata.getLong(1)
    deletes.getOrElseUpdate(file, scala.collection.mutable.ArrayBuffer.empty[Long]) += pos
    
    // The new record 'row' needs to be inserted.
    // In a full implementation, we'd also buffer `row` to write as a new Parquet file.
  }

  override def insert(row: InternalRow): Unit = {
    // In a full implementation, we'd buffer `row` to write as a new Parquet file.
  }

  override def commit(): WriterCommitMessage = {
    if (deletes.nonEmpty) {
      // Serialize deletes to JSON
      val mapper = new ObjectMapper()
      
      // Convert Scala Map to Java Map for Jackson
      val javaMap = new java.util.HashMap[String, java.util.List[Long]]()
      deletes.foreach { case (k, v) =>
        val list = new java.util.ArrayList[Long]()
        v.foreach(list.add)
        javaMap.put(k, list)
      }
      
      val jsonStr = mapper.writeValueAsString(javaMap)
      
      // Call Rust Core
      val jniBridge = com.hyperstreamdb.spark.jni.HyperStreamJNIBridge.getInstance()
      jniBridge.setGpuContext(gpuDevice)
      val success = jniBridge.commitPositionDeletes(tableName.split("\\.").last, jsonStr)
      // For now, print what would happen
      println(s"HyperStreamDB: Committing position deletes to Rust Core for $tableName")
      println(s"HyperStreamDB: Payload -> $jsonStr")
    }
    
    new HyperStreamDeltaCommitMessage("success")
  }

  override def abort(): Unit = {
    deletes.clear()
  }

  override def close(): Unit = {
    deletes.clear()
  }
}
