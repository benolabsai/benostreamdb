package com.benostreamdb.spark

import org.junit.Test
import org.junit.Assert._
import com.benostreamdb.spark.gpu.GpuContextResolver

class GpuContextResolverTest {

  @Test
  def testDefaultResolution(): Unit = {
    val dev = GpuContextResolver.resolveGpuDevice("auto")
    assertNotNull(dev)
    assertTrue(dev == "auto" || dev.startsWith("cuda:"))
  }

  @Test
  def testExplicitConfigResolution(): Unit = {
    assertEquals("cpu", GpuContextResolver.resolveGpuDevice("cpu"))
    assertEquals("cuda:1", GpuContextResolver.resolveGpuDevice("cuda:1"))
    assertEquals("mps", GpuContextResolver.resolveGpuDevice("mps"))
  }

  @Test
  def testBindTaskGpuContext(): Unit = {
    val dev = GpuContextResolver.bindTaskGpuContext("cpu")
    assertEquals("cpu", dev)
  }
}
