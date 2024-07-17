package it.unibo.alchemist.boundary.launchers

import ai.djl.nn.Block
import com.google.common.collect.Lists
import it.unibo.alchemist.boundary.{Launcher, Loader, Variable}
import it.unibo.alchemist.core.Simulation
import it.unibo.alchemist.model.Node
import it.unibo.alchemist.model.implementations.reactions.GlobalReactionStrategyExecutor
import it.unibo.alchemist.model.learning._
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.timedistributions.DiracComb
import it.unibo.alchemist.model.times.DoubleTime
import org.apache.commons.lang3.NotImplementedException
import org.apache.commons.math3.random.MersenneTwister
import org.slf4j.{Logger, LoggerFactory}

import java.util.concurrent.{ConcurrentLinkedQueue, Executors, TimeUnit}
import scala.collection.mutable
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.jdk.CollectionConverters.{IteratorHasAsScala, _}
import scala.util.{Failure, Success}

abstract class SwarMDPBaseLauncher(
    val batch: java.util.ArrayList[String],
    val globalRounds: Int,
    val parallelism: Int,
    val seedName: String,
    val strategies: List[ExecutionStrategy[Any, Nothing]],
    val globalSeed: Long,
    val globalBufferSize: Int
) extends Launcher {

  private val logger: Logger = LoggerFactory.getLogger(this.getClass.getName)
  private val errorQueue = new ConcurrentLinkedQueue[Throwable]()
  private val executor = Executors.newFixedThreadPool(parallelism)
  implicit private val executionContext: ExecutionContext = ExecutionContext.fromExecutor(executor)
  override def launch(loader: Loader): Unit = {
    val instances = loader.getVariables
    val prod = cartesianProduct(instances, batch)
    initializeNetwork()

    val experiences = List.fill(prod.size)(ExperienceBuffer[State](globalBufferSize))
    Range.inclusive(1, globalRounds).foreach { iter =>
      logger.info(s"Starting Global Round: $iter")
      logger.info(s"Number of simulations: ${prod.size}")
      val simulations = prod.zipWithIndex
        .to(LazyList)
        .map { case (instance, index) =>
          instance.addOne("globalRound" -> iter)
          val sim = loader.getWith[Any, Nothing](instance.asJava)
          val seed = instance(seedName).asInstanceOf[Double].toLong
          scheduleStrategies(strategies, seed, sim)
          println(s"${Thread.currentThread().getName}")
          neuralNetworkInjection(sim, iter - 1)
          runSimulationAsync(sim, index, instance)
        }
        .grouped(parallelism)

      val completedSimulations = simulations.flatMap { batch =>
        val result = Await.result(Future.sequence(batch), Duration.Inf)
        result
      }.toList
      val currentExperiences = collectExperience(completedSimulations)
      experiences.zip(currentExperiences).foreach { case (old, newest) => old.addAll(newest.getAll) }
      improvePolicy(experiences, iter - 1)
      cleanAfterRound(completedSimulations)
      strategies.foreach(_.progressRound)
    }

    saveNetworks()
    println("[DEBUG] finished ")
    // TODO - check executor shutdown
    executor.shutdown()
    executor.awaitTermination(Long.MaxValue, TimeUnit.DAYS)
    // TODO - throws errors in errorQueue
  }

  private def cartesianProduct(
      variables: java.util.Map[String, Variable[_]],
      variablesNames: java.util.List[String]
  ): List[mutable.Map[String, Serializable]] = {
    val l = variablesNames
      .stream()
      .map { variable =>
        val values = variables.get(variable)
        values.stream().map(e => variable -> e).toList
      }
      .toList
    Lists
      .cartesianProduct(l)
      .stream()
      .map(e => mutable.Map.from(e.iterator().asScala.toList))
      .iterator()
      .asScala
      .toList
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
      simulation.getError.ifPresent(error => throw error)
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

  protected def neuralNetworkInjection(simulation: Simulation[Any, Nothing], iteration: Int): Unit

  protected def initializeNetwork(): Unit

  protected def saveNetworks(): Unit

  private def collectExperience(simulations: List[Simulation[Any, Nothing]]): Seq[ExperienceBuffer[State]] = {
    simulations.map { simulation =>
      nodes(simulation)
        .map(_.getConcentration(new SimpleMolecule(Molecules.experience)))
        .map(_.asInstanceOf[ExperienceBuffer[State]])
        .map(_.getAll)
        .foldLeft(ExperienceBuffer[State](globalBufferSize)) { (buffer, experience) =>
          buffer.addAll(experience)
          buffer
        }
    }
  }
  protected def improvePolicy(simulationsExperience: Seq[ExperienceBuffer[State]], iteration: Int): Unit

  protected def loadNetworks(iteration: Int): (Block, Block)

  private def nodes(simulation: Simulation[Any, Nothing]): List[Node[Any]] =
    simulation.getEnvironment.getNodes.iterator().asScala.toList

  protected def scheduleStrategies(
      strategies: List[ExecutionStrategy[Any, Nothing]],
      seed: Long,
      simulation: Simulation[Any, Nothing]
  ): Unit = {
    simulation.schedule { () =>
      val random = new MersenneTwister(seed)
      strategies.zipWithIndex.foreach {
        case (strategy: GlobalExecution[Any, Nothing], index) =>
          val rate = (index + 1).toDouble / 10.0
          val timeDistribution = new DiracComb[Any](new DoubleTime(rate), 1)
          val environment = simulation.getEnvironment
          val reaction =
            new GlobalReactionStrategyExecutor[Any, Nothing](environment, random, timeDistribution, strategy)
          simulation.getEnvironment.addGlobalReaction(reaction)
        case (strategy: LocalExecution[Any, Nothing], index) =>
          throw new NotImplementedException("This feature has not been implemented yet!")
        case _ => throw new UnsupportedOperationException("Strategies can only be local or global!")
      }
    }
  }

  protected def cleanAfterRound(simulations: List[Simulation[Any, Nothing]]): Unit

}
