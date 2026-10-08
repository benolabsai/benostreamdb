package com.benostreamdb.spark

import com.fasterxml.jackson.databind.{JsonNode, ObjectMapper}

import scala.collection.JavaConverters._

/**
 * Minimal JSON helpers backing the `json_*` catalog functions. The engine
 * stores JSON as text and exposes PostgreSQL-compatible `json_*` functions;
 * these mirror that behaviour for Spark-side evaluation.
 */
object BenoStreamJson {

  private val mapper = new ObjectMapper()

  private def parse(json: String): Option[JsonNode] =
    try Option(mapper.readTree(json))
    catch { case _: Throwable => None }

  /** Split a dotted path (`a.b.c`) into segments. */
  private def segments(path: String): Seq[String] =
    path.split('.').map(_.trim).filter(_.nonEmpty)

  private def at(node: JsonNode, path: String): Option[JsonNode] = {
    var current: Option[JsonNode] = Some(node)
    for (seg <- segments(path) if current.isDefined) {
      val n = current.get
      current =
        if (n.isArray) {
          try Some(n.get(seg.toInt))
          catch { case _: Throwable => None }
        } else Option(n.get(seg))
    }
    current
  }

  /** Value at `path` rendered as JSON (null when absent). */
  def extractPath(json: String, path: String): String =
    parse(json).flatMap(at(_, path)).map(_.toString).orNull

  /** Value at `path` rendered as text (null when absent). */
  def extractPathText(json: String, path: String): String =
    parse(json).flatMap(at(_, path)).map(n => if (n.isTextual) n.asText() else n.toString).orNull

  /** Recursive containment: every field/element of `candidate` is in `json`. */
  def contains(json: String, candidate: String): Boolean = {
    def contained(haystack: JsonNode, needle: JsonNode): Boolean = {
      if (needle.isObject) {
        needle.fields().asScala.forall { e =>
          haystack.has(e.getKey) && contained(haystack.get(e.getKey), e.getValue)
        }
      } else if (needle.isArray) {
        needle.elements().asScala.forall { item =>
          haystack.isArray && haystack.elements().asScala.exists(contained(_, item))
        }
      } else {
        haystack == needle
      }
    }
    (parse(json), parse(candidate)) match {
      case (Some(h), Some(n)) => contained(h, n)
      case _ => false
    }
  }

  /** Whether a top-level key (or array element) exists. */
  def exists(json: String, key: String): Boolean =
    parse(json).exists { n =>
      if (n.isArray) {
        try n.get(key.toInt) != null
        catch { case _: Throwable => false }
      } else n.has(key)
    }

  /** JSON type name of the document. */
  def typeof(json: String): String =
    parse(json)
      .map { n =>
        if (n.isObject) "object"
        else if (n.isArray) "array"
        else if (n.isTextual) "string"
        else if (n.isNumber) "number"
        else if (n.isBoolean) "boolean"
        else if (n.isNull) "null"
        else "unknown"
      }
      .orNull

  /** Evaluate a jsonpath subset (`$.a.b`, `$.a[0]`, `$.a[*]`). */
  private def jsonPath(json: String, path: String): Seq[JsonNode] = {
    val cleaned = path.stripPrefix("$").stripPrefix(".")
    val parts = cleaned.split('.').map(_.trim).filter(_.nonEmpty)
    var nodes: Seq[JsonNode] = parse(json).toSeq
    for (part <- parts) {
      val (name, wildcard) = if (part.endsWith("[*]")) (part.stripSuffix("[*]"), true) else (part, false)
      val indexed: Option[(String, String)] = name match {
        case n if n.endsWith("]") && n.contains("[") =>
          val base = n.substring(0, n.indexOf('['))
          val idx = n.substring(n.indexOf('[') + 1, n.length - 1)
          Some((base, idx))
        case n => Some((n, ""))
      }
      nodes = nodes.flatMap { node =>
        indexed match {
          case Some((base, idx)) =>
            val target = if (base.isEmpty) Some(node) else Option(node.get(base))
            target.toSeq.flatMap { t =>
              if (wildcard) t.elements().asScala.toSeq
              else if (idx.nonEmpty) {
                try Option(t.get(idx.toInt)).toSeq
                catch { case _: Throwable => Seq.empty }
              } else Seq(t)
            }
          case None => Seq.empty
        }
      }
    }
    nodes
  }

  def pathExists(json: String, path: String): Boolean = jsonPath(json, path).nonEmpty

  /** Matches of a jsonpath as a JSON array string. */
  def pathQuery(json: String, path: String): String = {
    val matches = jsonPath(json, path)
    if (matches.isEmpty) null
    else mapper.createArrayNode().tap { arr => matches.foreach(arr.add) }.toString
  }

  private implicit class TapOps[A](private val a: A) extends AnyVal {
    def tap(f: A => Unit): A = { f(a); a }
  }
}
