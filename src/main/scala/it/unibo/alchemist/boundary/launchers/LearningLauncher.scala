package it.unibo.alchemist.boundary.launchers

import com.google.common.collect.Lists
import it.unibo.alchemist.boundary.{Launcher, Loader, Variable}
import it.unibo.alchemist.core.Simulation
import it.unibo.interop.PythonModules._
import it.unibo.alchemist.model.Node
import it.unibo.alchemist.model.learning.{Experience, ExperienceBuffer, Molecules, State}
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.util.BugReporting
import it.unibo.interop.PythonModules.pythonUtils
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.{PyQuote, SeqConverters}
import org.slf4j.{Logger, LoggerFactory}

import scala.jdk.CollectionConverters._
import java.util.concurrent.{ConcurrentLinkedQueue, Executors, TimeUnit}
import scala.collection.mutable
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.jdk.CollectionConverters.IteratorHasAsScala
import scala.util.{Failure, Success}

class LearningLauncher (
                         val batch: java.util.ArrayList[String],
                         val autoStart: Boolean,
                         val showProgress: Boolean,
                         val globalRounds: Int,
                         val seedName: String,
                         val miniBatchSize: Int
                       ) extends Launcher {

  private val parallelism: Int = Runtime.getRuntime.availableProcessors()
  private val logger: Logger = LoggerFactory.getLogger(this.getClass.getName)
  private val errorQueue = new ConcurrentLinkedQueue[Throwable]()
  private val executor = Executors.newFixedThreadPool(parallelism)
  private implicit val executionContext: ExecutionContext = ExecutionContext.fromExecutor(executor)

  override def launch(loader: Loader): Unit = {
    val instances = loader.getVariables
    val prod = cartesianProduct(instances, batch)

    Range.inclusive(1, globalRounds).foreach { iter =>
      logger.info(s"Starting Global Round: $iter")
      println(s"[AAAAAAAAAAAAAAAAAAAAAA] Starting Global Round: $iter")
      val futures = prod.zipWithIndex.map {
        case (instance, index) =>
          val sim = loader.getWith[Any, Nothing](instance.asJava)
          val seed = instance(seedName).asInstanceOf[Double]
          neuralNetworkInjection(sim, seed)
          runSimulationAsync(sim, index, instance)
      }
      Await
        .ready(Future.sequence(futures), Duration.Inf)
        .onComplete {
          case Success(simulations) =>
            val experience = collectExperience(simulations)
            improvePolicy(experience)
            cleanPythonObjects(simulations)
          case Failure(exception) =>
            println(exception)
            throw exception
        }
    }

    println("[AAAAAAAAAAA] finished ")
    // TODO - check executor shutdown
    executor.shutdown()
    executor.awaitTermination(Long.MaxValue, TimeUnit.DAYS)
    // TODO - throws errors in errorQueue
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

  private def runSimulationAsync(
    simulation: Simulation[Any, Nothing],
    index: Int,
    instance: mutable.Map[String, Serializable]
  )(implicit executionContext: ExecutionContext): Future[Simulation[Any, Nothing]] = {
    val future = Future {
      simulation.play()
      simulation.run()
      simulation.getError.ifPresent { error => throw error }
      logger.info("Simulation with {} completed successfully", instance)
      simulation
    }
    future.onComplete {
      case Success(_) =>
        logger.info("Simulation {} of {} completed", index + 1, instance.size)
      case Failure(exception) =>
        logger.error(s"Failure for simulation with $instance", exception)
        errorQueue.add(exception)
        executor.shutdownNow()
    }
    future
  }

  private def neuralNetworkInjection(simulation: Simulation[Any, Nothing], seed: Double): Unit = {
    val model = pythonUtils.load_neural_network(seed)
    nodes(simulation)
      .foreach { node =>
         node.setConcentration(new SimpleMolecule(Molecules.model), model)
      }
  }

  private def collectExperience(simulations: List[Simulation[Any, Nothing]]): Seq[ExperienceBuffer[State]] = {
    simulations.flatMap { simulation =>
      nodes(simulation)
        .map { node =>
          node.getConcentration(new SimpleMolecule(Molecules.experience)).asInstanceOf[ExperienceBuffer[State]]
        }
    }
  }

  private def improvePolicy(simulationsExperience: Seq[ExperienceBuffer[State]]): Unit = {
    // TODO - maybe this should be customizable with strategy or something similar
    simulationsExperience
      .foreach { buffer =>
        val iterations = Math.floor(buffer.size / miniBatchSize).toInt
        Range.inclusive(1, 2).foreach { iter =>
          val (actualStateBatch, actionBatch, rewardBatch, nextStateBatch) = toBatches(buffer.sample(miniBatchSize))
          // TODO - implement in python
          // pythonUtils.improve_policy(actualStateBatch, actionBatch, rewardBatch, nextStateBatch)
        }
      }
  }

  private def toBatches(experience: Seq[Experience[State]]): (py.Dynamic, py.Dynamic, py.Dynamic, py.Dynamic)= {
    val encodedBuffer = experience.map(_.encode)
    val actualStateBatch = torch.Tensor(encodedBuffer.map(_._1.toPythonCopy).toPythonCopy)
    val actionBatch = torch.Tensor(encodedBuffer.map(_._2).toPythonCopy)
    val rewardBatch = torch.Tensor(encodedBuffer.map(_._3).toPythonCopy)
    val nextStateBatch = torch.Tensor(encodedBuffer.map(_._4.toPythonCopy).toPythonCopy)
    (actualStateBatch, actionBatch, rewardBatch, nextStateBatch)
  }

  private def nodes(simulation: Simulation[Any, Nothing]): List[Node[Any]] = {
    simulation.getEnvironment.getNodes.iterator().asScala.toList
  }

  private def cleanPythonObjects(simulations: List[Simulation[Any, Nothing]]): Unit = {
    val gc = py.module("gc")
    simulations.foreach { simulation =>
      try {
        nodes(simulation).foreach { node =>
          node.getConcentration(new SimpleMolecule(Molecules.model)).asInstanceOf[py.Dynamic].del()
        }
        gc.collect()
        Runtime.getRuntime.gc()
      } catch {
        case e: Throwable => println(e)
      }
    }
  }

}