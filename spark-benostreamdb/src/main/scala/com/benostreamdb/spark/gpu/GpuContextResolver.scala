package com.benostreamdb.spark.gpu

import org.apache.spark.TaskContext
import org.slf4j.LoggerFactory
import com.benostreamdb.spark.jni.BenoStreamJNIBridge

object GpuContextResolver {
  private val logger = LoggerFactory.getLogger(GpuContextResolver.getClass)

  /**
   * Resolves the appropriate GPU device string for the currently executing Spark task.
   *
   * Priority:
   * 1. Dynamic TaskContext GPU resource allocation (e.g. spark.task.resource.gpu.amount)
   * 2. CUDA_VISIBLE_DEVICES environment variable
   * 3. Configured fallback (e.g. table property or spark.benostream.gpu.device)
   * 4. "auto"
   */
  def resolveGpuDevice(configuredDevice: String = "auto"): String = {
    // 1. Check TaskContext GPU resources if executing within a task
    val taskGpu = Option(TaskContext.get()).flatMap { ctx =>
      try {
        val resources = ctx.resources()
        if (resources != null && resources.contains("gpu")) {
          val gpuInfo = resources("gpu")
          val addresses = gpuInfo.addresses
          if (addresses != null && addresses.nonEmpty) {
            Some(s"cuda:${addresses.head}")
          } else None
        } else None
      } catch {
        case _: Throwable => None
      }
    }

    if (taskGpu.isDefined) {
      return taskGpu.get
    }

    // 2. Check if explicitly configured to a specific device (e.g. "cuda:0", "cpu", "mps")
    if (configuredDevice != null && configuredDevice.nonEmpty && configuredDevice != "auto") {
      return configuredDevice
    }

    // 3. Check environment variable CUDA_VISIBLE_DEVICES
    sys.env.get("CUDA_VISIBLE_DEVICES").filter(_.trim.nonEmpty) match {
      case Some(dev) =>
        val firstDev = dev.split(",").head.trim
        s"cuda:$firstDev"
      case None =>
        "auto"
    }
  }

  /**
   * Binds the thread's native GPU context to the resolved device for this task.
   */
  def bindTaskGpuContext(configuredDevice: String = "auto"): String = {
    val device = resolveGpuDevice(configuredDevice)
    if (BenoStreamJNIBridge.isLoaded) {
      try {
        val jni = BenoStreamJNIBridge.getInstance()
        jni.setGpuContext(device)
      } catch {
        case e: UnsatisfiedLinkError =>
          logger.warn(s"Native setGpuContext not linked, skipping GPU binding: ${e.getMessage}")
      }
    }
    device
  }
}
