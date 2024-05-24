package it.unibo.alchemist.boundary.launchers

import com.google.common.collect.Lists
import it.unibo.alchemist.boundary.launchers.LearningLauncher.logger
import it.unibo.alchemist.boundary.{Launcher, Loader, Variable}
import it.unibo.alchemist.core.Simulation
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.util.BugReporting
import org.slf4j.{Logger, LoggerFactory}

import scala.jdk.CollectionConverters._
import java.util.concurrent.{ConcurrentLinkedQueue, Executors}
import java.util.concurrent.atomic.AtomicInteger
import scala.collection.mutable
import scala.jdk.CollectionConverters.IteratorHasAsScala

class LearningLauncher (
                         val batch: java.util.ArrayList[String],
                         val autoStart: Boolean,
                         val showProgress: Boolean,
                         val globalRounds: Int
                       ) extends Launcher {

  private val parallelism: Int = Runtime.getRuntime.availableProcessors()
  private val logger: Logger = LoggerFactory.getLogger(this.getClass.getName)

  override def launch(loader: Loader): Unit = {
    val instances = loader.getVariables
    val prod = cartesianProduct(instances, batch)

    val workerId = new AtomicInteger(0)
    val errorQueue = new ConcurrentLinkedQueue[Throwable]()

    Range.inclusive(1, globalRounds).foreach { iter =>
      logger.info(s"---------------- Global Round $iter ----------------")
      prod.zipWithIndex.foreach {
        case (instance, index) =>
          val sim = loader.getWith[Any, Nothing](instance.asJava)
          neuralNetworkInjection(sim)
          sim.play()
          sim.run()
          logger.info("Simulation with {} completed successfully", instance)
      }
    }

  }

  private def cartesianProduct(
    variables: java.util.Map[String, Variable[_]],
    variablesNames: java.util.List[String]
  ): List[mutable.Map[String, Serializable]] = {
    val l = variablesNames.stream().map(
      variable => {
        val values = variables.get(variable)
        values.stream().map(e => variable -> e).toList
      }).toList
    Lists.cartesianProduct(l)
      .stream()
      .map(e => { mutable.Map.from(e.iterator().asScala.toList) })
      .iterator().asScala.toList
      .asInstanceOf[List[mutable.Map[String, Serializable]]]
  }

 private def neuralNetworkInjection(simulation: Simulation[Any, Nothing]): Unit = {
   simulation
     .getEnvironment
     .getNodes
     .iterator()
     .asScala.toList
     .foreach { node =>
       node.setConcentration(new SimpleMolecule("Model"), 2) // TODO - set real model with scalapy
     }
 }

}