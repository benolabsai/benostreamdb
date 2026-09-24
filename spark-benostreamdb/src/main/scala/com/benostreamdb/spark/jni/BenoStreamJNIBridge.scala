package com.benostreamdb.spark.jni

import org.slf4j.LoggerFactory

class BenoStreamJNIBridge private () {
  
  @native def addIndex(table: String, column: String, indexType: String): Boolean
  @native def buildIndex(table: String, segmentId: String): Boolean
  @native def setPrimaryKey(table: String, columns: String): Boolean
  @native def queryIndexIn(table: String, column: String, valuesJson: String): String
  @native def commitPositionDeletes(table: String, deletesJson: String): Boolean
  @native def setGpuContext(deviceType: String): Boolean
  @native def vectorSearch(table: String, segmentId: String, column: String, k: Int, queryVectorPtr: Long, queryVectorLen: Int, outArrayPtr: Long, outSchemaPtr: Long): Int

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
