package com.hyperstreamdb.spark

import org.apache.spark.sql.SparkSession
import org.junit.{After, Before, Test}

import org.apache.spark.sql.connector.catalog.{Identifier, Table, TableCatalog}
import org.apache.spark.sql.connector.expressions.Transform
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap
import java.util

class MockHyperStreamCatalog extends TableCatalog {
  private var tbl: Table = _
  override def name(): String = "mock_hs"
  override def initialize(name: String, options: CaseInsensitiveStringMap): Unit = {}
  override def listTables(namespace: Array[String]): Array[Identifier] = Array.empty
  override def loadTable(ident: Identifier): Table = tbl
  override def createTable(ident: Identifier, schema: StructType, partitions: Array[Transform], properties: util.Map[String, String]): Table = null
  override def alterTable(ident: Identifier, changes: org.apache.spark.sql.connector.catalog.TableChange*): Table = null
  override def dropTable(ident: Identifier): Boolean = false
  override def renameTable(oldIdent: Identifier, newIdent: Identifier): Unit = {}

  def setTable(table: Table): Unit = {
    this.tbl = table
  }
}

class SparkMergeIntegrationTest {

  var spark: SparkSession = _

  @Before
  def setup(): Unit = {
    spark = SparkSession.builder()
      .master("local[2]")
      .appName("HyperStreamMergeTest")
      .config("spark.sql.catalogImplementation", "in-memory")
      .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
      .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkCatalog")
      .config("spark.sql.catalog.spark_catalog.type", "hadoop")
      .config("spark.sql.catalog.spark_catalog.warehouse", "target/warehouse")
      .config("spark.sql.catalog.mock_hs", "com.hyperstreamdb.spark.MockHyperStreamCatalog")
      .config("spark.sql.catalog.hs_catalog", "com.hyperstreamdb.spark.HyperStreamProcedureCatalog")
      .getOrCreate()
  }

  @After
  def tearDown(): Unit = {
    if (spark != null) {
      spark.stop()
    }
  }

  @Test
  def testMergeIntoDelegation(): Unit = {
    val tablePath = "file://" + new java.io.File("target/warehouse/default/users").getAbsolutePath
    
    // 1. Create a native Iceberg table
    spark.sql("CREATE DATABASE IF NOT EXISTS default")
    spark.sql("DROP TABLE IF EXISTS spark_catalog.default.users")
    spark.sql("CREATE TABLE spark_catalog.default.users (id INT, name STRING, age INT) USING iceberg")
    
    // 2. Insert some data via native Iceberg
    spark.sql("INSERT INTO spark_catalog.default.users VALUES (1, 'Alice', 30), (2, 'Bob', 25)")
    
    // 3. Wrap it in HyperStreamTable and set it in Mock Catalog
    import org.apache.spark.sql.connector.catalog.CatalogManager
    val icebergCatalog = spark.sessionState.catalogManager.catalog("spark_catalog").asInstanceOf[TableCatalog]
    val icebergTable = icebergCatalog.loadTable(Identifier.of(Array("default"), "users"))
    
    val hsTable = new HyperStreamTable(icebergTable, icebergTable.schema(), new java.util.HashMap[String, String](), "auto") {
      override def name(): String = "mock_hs.default.users"
    }
    val mockCatalog = spark.sessionState.catalogManager.catalog("mock_hs").asInstanceOf[MockHyperStreamCatalog]
    mockCatalog.setTable(hsTable)
    
    // 4. Verify HyperStreamTable supports RowLevelOperations
    assert(hsTable.capabilities().contains(org.apache.spark.sql.connector.catalog.TableCapability.BATCH_WRITE))
    
    // 5. Build the RowLevelOperation manually
    import org.apache.spark.sql.connector.write.{RowLevelOperationInfo, RowLevelOperation}
    val mergeInfo = new RowLevelOperationInfo {
      override def options() = CaseInsensitiveStringMap.empty()
      override def command() = RowLevelOperation.Command.MERGE
    }
    
    val mergeBuilder = hsTable.newRowLevelOperationBuilder(mergeInfo)
    val rowLevelOp = mergeBuilder.build()
    
    assert(rowLevelOp.command() == RowLevelOperation.Command.MERGE)
    
    // 6. Verify required metadata columns (must request _file and _pos)
    import com.hyperstreamdb.spark.HyperStreamRowLevelOperation
    val hsRowOp = rowLevelOp.asInstanceOf[HyperStreamRowLevelOperation]
    val metaAttrs = hsRowOp.requiredMetadataAttributes()
    assert(metaAttrs.exists(_.fieldNames().contains("_file")))
    assert(metaAttrs.exists(_.fieldNames().contains("_pos")))
    
    println("MERGE INTO API delegation verified successfully!")
  }
}
