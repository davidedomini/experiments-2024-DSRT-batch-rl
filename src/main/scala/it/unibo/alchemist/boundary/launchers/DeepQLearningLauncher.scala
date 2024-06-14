package it.unibo.alchemist.boundary.launchers

import com.google.common.collect.Lists
import it.unibo.alchemist.boundary.launchers.DeepQLearningLauncher.DQNFactory
import it.unibo.alchemist.boundary.{Launcher, Loader, Variable}
import it.unibo.alchemist.core.Simulation
import it.unibo.alchemist.model.implementations.reactions.{AbstractGlobalReaction, GlobalReactionStrategyExecutor}
import it.unibo.alchemist.model.layers.ModelLayer
import it.unibo.interop.PythonModules._
import it.unibo.alchemist.model.{Environment, Layer, Node, Position, Time}
import it.unibo.alchemist.model.learning.{
  Action,
  ExecutionStrategy,
  Experience,
  ExperienceBuffer,
  GlobalExecution,
  LocalExecution,
  Molecules,
  State
}
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.timedistributions.DiracComb
import it.unibo.alchemist.model.times.DoubleTime
import it.unibo.alchemist.util.BugReporting
import it.unibo.experiment.{ActionSpace, SimpleSequentialDQN}
import it.unibo.interop.PythonModules.pythonUtils
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.{PyQuote, SeqConverters}
import org.apache.commons.lang3.NotImplementedException
import org.apache.commons.math3.random.MersenneTwister
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

class DeepQLearningLauncher(
    batch: java.util.ArrayList[String],
    globalRounds: Int,
    parallelism: Int,
    seedName: String,
    strategies: List[ExecutionStrategy[Any, Nothing]],
    globalSeed: Long,
    globalBufferSize: Int,
    learningInfo: DeepQLearningLauncher.LearningInfo,
    networkFactory: DQNFactory
) extends SwarMDPBaseLauncher(batch, globalRounds, parallelism, seedName, strategies, globalSeed, globalBufferSize) {
  torch.manual_seed(globalSeed)
  import learningInfo._
  private var models: List[py.Dynamic] = List.empty
  private var targets: List[py.Dynamic] = List.empty

  protected def neuralNetworkInjection(simulation: Simulation[Any, Nothing], iteration: Int): Unit = {
    val (model, _) = loadNetworks(iteration)
    val layer = new ModelLayer[Any, Nothing](simulation.getEnvironment, model)
    simulation.getEnvironment.addLayer(new SimpleMolecule(Molecules.model), layer)
  }

  protected def initializeNetwork(): Unit = {
    val network = networkFactory.create()
    models = models :+ network
    targets = targets :+ network
  }

  protected def saveNetworks(): Unit = {
    val path = "networks-snapshots/"
    Files.createDirectories(Paths.get(path))
    models.zipWithIndex.foreach { case (model, index) =>
      torch.save(model.state_dict(), s"${path}network-iteration-$index")
    }
  }

  private var ticks = 0
  protected def improvePolicy(simulationsExperience: Seq[ExperienceBuffer[State]], iteration: Int): Unit = {
    // TODO - maybe this should be customizable with strategy or something similar
    println(s"Loading nn Iteration $iteration")
    val (actionNetwork, targetNetwork) = loadNetworks(iteration)
    val optimizer = torch.optim.Adam(actionNetwork.parameters(), learningRate)
    val allSize = simulationsExperience.map(_.getAll.size).sum
    val mergedBuffer = simulationsExperience.foldLeft(ExperienceBuffer[State](allSize)) { (buffer, experience) =>
      buffer.addAll(experience.getAll)
      buffer
    }
    val iterations = (mergedBuffer.getAll.size / miniBatchSize) / 5.0
    Range.inclusive(1, Math.min(iterations.toInt, learningInfo.iterations)).foreach { iter =>
      ticks += 1
      val (actualStateBatch, actionBatch, rewardBatch, nextStateBatch) = toBatches(mergedBuffer.sample(miniBatchSize))
      val networkPass =
        actionNetwork(torch.tensor(actualStateBatch))
      val stateActionValue = networkPass.gather(1, actionBatch.view(miniBatchSize, 1))
      val nextStateValues = targetNetwork(torch.tensor(nextStateBatch)).max(1).bracketAccess(0).detach()
      val expectedValue = ((nextStateValues * gamma) + rewardBatch).detach()
      val criterion = torch.nn.SmoothL1Loss()
      val logsumexp = torch.logsumexp(networkPass, 1, keepdim = true)
      val cqlLoss = (logsumexp - stateActionValue).mean()
      val loss = criterion(stateActionValue, expectedValue.unsqueeze(1))

      val totalLoss = loss //+ (cqlLoss * 0.5)

      optimizer.zero_grad()
      totalLoss.backward()
      torch.nn.utils.clip_grad_value_(actionNetwork.parameters(), 1.0)
      optimizer.step()
      if (ticks % updateEach == 0) {
        targetNetwork.load_state_dict(actionNetwork.state_dict())
      }

      models = models :+ actionNetwork
      targets = targets :+ targetNetwork
    }
  }

  protected def loadNetworks(iteration: Int): (py.Dynamic, py.Dynamic) = {
    val actionNetwork = networkFactory.create()
    val targetNetwork = networkFactory.create()
    val model = models(iteration)
    val target = targets(iteration)
    actionNetwork.load_state_dict(model.state_dict())
    targetNetwork.load_state_dict(target.state_dict())
    (actionNetwork, targetNetwork)
  }

  private def toBatches(experience: Seq[Experience[State]]): (py.Dynamic, py.Dynamic, py.Dynamic, py.Dynamic) = {
    val encodedBuffer = experience.map(_.encode)
    val actualStateBatch = torch.tensor(encodedBuffer.map(_._1.toPythonCopy).toPythonCopy)
    val actionBatch = torch.tensor(encodedBuffer.map(_._2).toPythonCopy, dtype = torch.int64)
    val rewardBatch = torch.tensor(encodedBuffer.map(_._3).toPythonCopy)
    val nextStateBatch = torch.tensor(encodedBuffer.map(_._4.toPythonCopy).toPythonCopy)
    (actualStateBatch, actionBatch, rewardBatch, nextStateBatch)
  }

  private def nodes(simulation: Simulation[Any, Nothing]): List[Node[Any]] =
    simulation.getEnvironment.getNodes.iterator().asScala.toList

  protected def cleanAfterRound(simulations: List[Simulation[Any, Nothing]]): Unit =
    cleanPythonObjects(simulations)

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

object DeepQLearningLauncher {
  case class LearningInfo(
      iterations: Int = 300,
      updateEach: Int = 4,
      gamma: Double = 0.9,
      learningRate: Double = 0.0005,
      miniBatchSize: Int = 64
  )

  class DQNFactory(input: Int, hidden: Int, output: Int) {
    def create(): py.Dynamic = SimpleSequentialDQN(input, hidden, output)
  }
}
