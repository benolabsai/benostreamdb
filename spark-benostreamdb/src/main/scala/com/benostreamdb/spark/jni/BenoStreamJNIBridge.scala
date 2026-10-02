package com.benostreamdb.spark.jni

import org.slf4j.LoggerFactory

class BenoStreamJNIBridge private () {
  
  @native def addIndex(table: String, column: String, indexType: String): Boolean
  @native def buildIndex(table: String, segmentId: String): Boolean
  @native def setPrimaryKey(table: String, columns: String): Boolean
  @native def queryIndexIn(table: String, column: String, valuesJson: String): String
  @native def commitPositionDeletes(table: String, deletesJson: String): Boolean
  @native def setGpuContext(deviceType: String): Boolean
  @native def dropIndex(table: String, column: String, indexType: String): Boolean
  @native def compactTable(table: String): Boolean
  @native def listIndexes(table: String): String
  @native def vectorSearch(table: String, segmentId: String, column: String, k: Int, queryVectorPtr: Long, queryVectorLen: Int, outArrayPtr: Long, outSchemaPtr: Long): Int
  @native def regionalDriftSearch(table: String, query: String, seedsJson: String, topK: Int, hops: Int, nDepth: Int, kFollowups: Int, mode: String, outArrayPtr: Long, outSchemaPtr: Long): Int
  @native def gatherMetrics(): String

  def runRegionalDriftSearch(
      table: String,
      query: String,
      seeds: Seq[Long],
      topK: Int,
      hops: Int,
      nDepth: Int,
      kFollowups: Int,
      mode: String
  ): Seq[Long] = {
    if (!BenoStreamJNIBridge.isLoaded || seeds.isEmpty) {
      return Seq.empty
    }

    import org.apache.arrow.c.{ArrowArray, ArrowSchema, Data}
    import org.apache.arrow.memory.RootAllocator
    import org.apache.arrow.vector.BigIntVector

    val allocator = new RootAllocator()
    val arrowArray = ArrowArray.allocateNew(allocator)
    val arrowSchema = ArrowSchema.allocateNew(allocator)
    
    val seedsJson = seeds.mkString("[\"", "\",\"", "\"]")

    try {
      val res = regionalDriftSearch(
        table, query, seedsJson, topK, hops, nDepth, kFollowups, mode,
        arrowArray.memoryAddress(), arrowSchema.memoryAddress()
      )
      if (res < 0) return Seq.empty

      val root = Data.importVectorSchemaRoot(allocator, arrowArray, arrowSchema, null)
      try {
        val rowIdVec = root.getVector("node_id").asInstanceOf[BigIntVector]
        val count = root.getRowCount
        val results = new scala.collection.mutable.ArrayBuffer[Long](count)
        var i = 0
        while (i < count) {
          results += rowIdVec.get(i)
          i += 1
        }
        results.toSeq
      } finally {
        root.close()
      }
    } catch {
      case e: Throwable =>
        BenoStreamJNIBridge.logger.error("Error invoking native regionalDriftSearch", e)
        Seq.empty
    } finally {
      arrowArray.close()
      arrowSchema.close()
      allocator.close()
    }
  }

  /**
   * High-level accelerated vector search using Arrow C Data interface.
   * If native library is loaded and index exists, executes accelerated HNSW search.
   * Otherwise returns empty sequence.
   */
  def searchVectors(
      table: String,
      segmentId: String,
      column: String,
      queryVector: Array[Float],
      k: Int
  ): Seq[(Long, Float)] = {
    if (!BenoStreamJNIBridge.isLoaded || queryVector == null || queryVector.length == 0 || k <= 0) {
      return Seq.empty
    }

    import org.apache.arrow.c.{ArrowArray, ArrowSchema, Data}
    import org.apache.arrow.memory.RootAllocator
    import org.apache.arrow.vector.{BigIntVector, Float4Vector}
    import java.nio.{ByteBuffer, ByteOrder}

    val allocator = new RootAllocator()
    val arrowArray = ArrowArray.allocateNew(allocator)
    val arrowSchema = ArrowSchema.allocateNew(allocator)

    // Allocate direct buffer for float query vector
    val directBuf = ByteBuffer.allocateDirect(queryVector.length * 4).order(ByteOrder.nativeOrder())
    directBuf.asFloatBuffer().put(queryVector)
    val queryPtr = try {
      val field = directBuf.getClass.getDeclaredField("address")
      field.setAccessible(true)
      field.getLong(directBuf)
    } catch {
      case _: Throwable => 0L
    }

    if (queryPtr == 0L) {
      arrowArray.close()
      arrowSchema.close()
      allocator.close()
      return Seq.empty
    }

    try {
      val res = vectorSearch(
        table,
        segmentId,
        column,
        k,
        queryPtr,
        queryVector.length,
        arrowArray.memoryAddress(),
        arrowSchema.memoryAddress()
      )
      if (res < 0) {
        return Seq.empty
      }

      val root = Data.importVectorSchemaRoot(allocator, arrowArray, arrowSchema, null)
      try {
        val rowIdVec = root.getVector("_row_id").asInstanceOf[BigIntVector]
        val distVec = root.getVector("_distance").asInstanceOf[Float4Vector]
        val count = root.getRowCount
        val results = new scala.collection.mutable.ArrayBuffer[(Long, Float)](count)
        var i = 0
        while (i < count) {
          results += ((rowIdVec.get(i), distVec.get(i)))
          i += 1
        }
        results.toSeq
      } finally {
        root.close()
      }
    } catch {
      case e: Throwable =>
        BenoStreamJNIBridge.logger.error("Error invoking native vectorSearch", e)
        Seq.empty
    } finally {
      arrowArray.close()
      arrowSchema.close()
      allocator.close()
    }
  }

}

object BenoStreamJNIBridge {
  private val logger = LoggerFactory.getLogger(classOf[BenoStreamJNIBridge])
  private var instance: BenoStreamJNIBridge = _
  private var loaded = false

  def getInstance(): BenoStreamJNIBridge = {
    if (instance == null) {
      synchronized {
        if (instance == null) {
          try {
            System.loadLibrary("benostreamdb")
            loaded = true
            logger.info("Successfully loaded native BenoStreamDB library.")
          } catch {
            case e: UnsatisfiedLinkError =>
              logger.warn(s"Failed to load native BenoStreamDB library: ${e.getMessage}. Using fallback/mock implementation for testing.")
          }
          instance = new BenoStreamJNIBridge()
        }
      }
    }
    instance
  }
  
  def isLoaded: Boolean = loaded
}
