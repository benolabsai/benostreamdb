package com.hyperstreamdb.spark.jni

import org.slf4j.LoggerFactory

class HyperStreamJNIBridge private () {
  
  @native def addIndex(table: String, column: String, indexType: String): Boolean
  @native def buildIndex(table: String, segmentId: String): Boolean
  @native def setPrimaryKey(table: String, columns: String): Boolean
  @native def queryIndexIn(table: String, column: String, valuesJson: String): String
  @native def commitPositionDeletes(table: String, deletesJson: String): Boolean
  @native def setGpuContext(deviceType: String): Boolean

}

object HyperStreamJNIBridge {
  private val logger = LoggerFactory.getLogger(classOf[HyperStreamJNIBridge])
  private var instance: HyperStreamJNIBridge = _
  private var loaded = false

  def getInstance(): HyperStreamJNIBridge = {
    if (instance == null) {
      synchronized {
        if (instance == null) {
          try {
            System.loadLibrary("hyperstreamdb")
            loaded = true
            logger.info("Successfully loaded native HyperStreamDB library.")
          } catch {
            case e: UnsatisfiedLinkError =>
              logger.warn(s"Failed to load native HyperStreamDB library: ${e.getMessage}. Using fallback/mock implementation for testing.")
          }
          instance = new HyperStreamJNIBridge()
        }
      }
    }
    instance
  }
  
  def isLoaded: Boolean = loaded
}
