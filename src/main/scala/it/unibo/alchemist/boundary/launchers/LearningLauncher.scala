package it.unibo.alchemist.boundary.launchers

import com.google.common.collect.Lists
import it.unibo.alchemist.boundary.{Launcher, Loader, Variable}
import it.unibo.alchemist.core.Simulation
import it.unibo.alchemist.model.implementations.reactions.{AbstractGlobalReaction, GlobalReactionStrategyExecutor}
import it.unibo.alchemist.model.layers.ModelLayer
import it.unibo.interop.PythonModules._
import it.unibo.alchemist.model.{Environment, Layer, Node, Position, Time}
import it.unibo.alchemist.model.learning.{ExecutionStrategy, Experience, ExperienceBuffer, GlobalExecution, LocalExecution, Molecules, State}
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.timedistributions.DiracComb
import it.unibo.alchemist.model.times.DoubleTime
import it.unibo.alchemist.util.BugReporting
import it.unibo.experiment.SimpleSequentialDQN
import it.unibo.interop.PythonModules.pythonUtils
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.{PyQuote, SeqConverters}
import org.apache.commons.lang3.NotImplementedException
import org.jooq.lambda.fi.lang.CheckedRunnable
import org.slf4j.{Logger, LoggerFactory}

import java.nio.file.{Files, Paths}
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
                         val miniBatchSize: Int,
                         val strategies: List[ExecutionStrategy[Any, Nothing]]
                       ) extends Launcher {

  private val parallelism: Int = Runtime.getRuntime.availableProcessors()
  private val logger: Logger = LoggerFactory.getLogger(this.getClass.getName)
  private val errorQueue = new ConcurrentLinkedQueue[Throwable]()
  private val executor = Executors.newFixedThreadPool(parallelism)
  private implicit val executionContext: ExecutionContext = ExecutionContext.fromExecutor(executor)

  override def launch(loader: Loader): Unit = {
    val instances = loader.getVariables
    val prod = cartesianProduct(instances, batch)
    initNN()

    Range.inclusive(1, globalRounds).foreach { iter =>
      logger.info(s"Starting Global Round: $iter")
      println(s"[DEBUG] Starting Global Round: $iter")
      val futures = prod.zipWithIndex.map {
        case (instance, index) =>
          val sim = loader.getWith[Any, Nothing](instance.asJava)
          val seed = instance(seedName).asInstanceOf[Double]
          scheduleStrategies(strategies, sim)
          neuralNetworkInjection(sim, iter-1)
          runSimulationAsync(sim, index, instance)
      }
      Await
        .ready(Future.sequence(futures), Duration.Inf)
        .onComplete {
          case Success(simulations) =>
            val experience = collectExperience(simulations)
            improvePolicy(experience, iter - 1)
            cleanPythonObjects(simulations)
          case Failure(exception) =>
            println(exception)
            throw exception
        }
    }

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

  private def neuralNetworkInjection(simulation: Simulation[Any, Nothing], iteration: Int): Unit = {
    val (model, _) = loadNetworks(iteration)
    //val layer = new ModelLayer[Any, Nothing](simulation.getEnvironment, model)
    //simulation.getEnvironment.addLayer(new SimpleMolecule(Molecules.model), layer)
    nodes(simulation) // TODO - parametrize
      .foreach { node =>
         node.setConcentration(new SimpleMolecule(Molecules.model), model)
      }
  }

  private def initNN(): Unit = {
    val path = "networks-snapshots/"
    Files.createDirectories(Paths.get(path))
    val network = SimpleSequentialDQN(10, 64, 8)
    torch.save(network.state_dict(), s"${path}network-iteration-0")
  }

  private def collectExperience(simulations: List[Simulation[Any, Nothing]]): Seq[ExperienceBuffer[State]] = {
    simulations.flatMap { simulation =>
      nodes(simulation)
        .map { node =>
          node.getConcentration(new SimpleMolecule(Molecules.experience)).asInstanceOf[ExperienceBuffer[State]]
        }
    }
  }

  private val targetNetworkUpdateRate = 2 // TODO - refactor
  private val gamma = 0.9
  private val learningRate = 0.0005
  private def improvePolicy(simulationsExperience: Seq[ExperienceBuffer[State]], iteration: Int): Unit = {
    // TODO - maybe this should be customizable with strategy or something similar
    println(s"Loading nn Iteration $iteration")
    val (actionNetwork, targetNetwork) = loadNetworks(iteration)
    val optimizer = torch.optim.RMSprop(actionNetwork.parameters(), learningRate)
    simulationsExperience
      .foreach { buffer =>
        val iterations = Math.floor(buffer.size / miniBatchSize).toInt
        Range.inclusive(1, iterations).foreach { iter =>
          val (actualStateBatch, actionBatch, rewardBatch, nextStateBatch) = toBatches(buffer.sample(miniBatchSize))
          val stateActionValue = actionNetwork(torch.tensor(actualStateBatch)).gather(1, actionBatch.view(miniBatchSize, 1))
          val nextStateValues = targetNetwork(torch.tensor(nextStateBatch)).max(1).bracketAccess(0).detach()
          val expectedValue = (nextStateValues * gamma) + rewardBatch
          val criterion = torch.nn.SmoothL1Loss()
          val loss = criterion(stateActionValue, expectedValue.unsqueeze(1))
          optimizer.zero_grad()
          loss.backward()
          torch.nn.utils.clip_grad_value_(actionNetwork.parameters(), 1.0)
          optimizer.step()
          if(iter % targetNetworkUpdateRate ==  0) {
            targetNetwork.load_state_dict(actionNetwork.state_dict())
          }
        }
      }
    torch.save(actionNetwork.state_dict(), s"networks-snapshots/network-iteration-${iteration+1}")
  }

  private def loadNetworks(iteration: Int): (py.Dynamic, py.Dynamic) = {
    val network = SimpleSequentialDQN(10, 64, 8)
    network.load_state_dict(torch.load(s"networks-snapshots/network-iteration-$iteration"))
    (network, network)
  }

  private def toBatches(experience: Seq[Experience[State]]): (py.Dynamic, py.Dynamic, py.Dynamic, py.Dynamic)= {
    val encodedBuffer = experience.map(_.encode)
    val actualStateBatch = torch.tensor(encodedBuffer.map(_._1.toPythonCopy).toPythonCopy)
    val actionBatch = torch.tensor(encodedBuffer.map(_._2).toPythonCopy, dtype=torch.int64)
    val rewardBatch = torch.tensor(encodedBuffer.map(_._3).toPythonCopy)
    val nextStateBatch = torch.tensor(encodedBuffer.map(_._4.toPythonCopy).toPythonCopy)
    (actualStateBatch, actionBatch, rewardBatch, nextStateBatch)
  }

  private def nodes(simulation: Simulation[Any, Nothing]): List[Node[Any]] = {
    simulation.getEnvironment.getNodes.iterator().asScala.toList
  }

  private def scheduleStrategies(strategies: List[ExecutionStrategy[Any, Nothing]], simulation: Simulation[Any, Nothing]): Unit = {
    simulation.schedule(() => {
      strategies.zipWithIndex.foreach {
        case (strategy: GlobalExecution[Any, Nothing], index) =>
          val rate = (index + 1).toDouble / 10.0
          val timeDistribution = new DiracComb[Any](new DoubleTime(rate), 1)
          val environment = simulation.getEnvironment
          val reaction = new GlobalReactionStrategyExecutor[Any, Nothing](environment, timeDistribution, strategy)
          simulation.getEnvironment.addGlobalReaction(reaction)
        case (strategy: LocalExecution[Any, Nothing], index) =>
          throw new NotImplementedException("This feature has not been implemented yet!")
        case _ => throw new UnsupportedOperationException("Strategies can only be local or global!")
      }
    })
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
