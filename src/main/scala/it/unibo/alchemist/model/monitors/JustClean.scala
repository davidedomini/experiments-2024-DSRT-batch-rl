package it.unibo.alchemist.model.monitors

import me.shadaj.scalapy.py
import it.unibo.alchemist.boundary.OutputMonitor
import it.unibo.alchemist.model.molecules.SimpleMolecule

import scala.jdk.CollectionConverters.IteratorHasAsScala
import it.unibo.alchemist.model.{Environment, Position, Time}

class JustClean[P <: Position[P]](seed: Double) extends OutputMonitor[Any, P] {

  override def finished(environment: Environment[Any, P], time: Time, step: Long): Unit =
    cleanPythonObjects(environment)

  private def cleanPythonObjects(environment: Environment[_, P]): Unit = {
    val gc = py.module("gc")
    try {
      val nodes = environment.getNodes.iterator().asScala.toList
      nodes.foreach { node =>
        node.getConcentration(new SimpleMolecule("Model")).asInstanceOf[py.Dynamic].del()
      }
      gc.collect()
      Runtime.getRuntime.gc()
    } catch {
      case e: Throwable => println(e)
    }
  }
  
}
