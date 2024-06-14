package it.unibo.alchemist.boundary.launchers

import com.google.common.collect.Lists
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

class LearningLauncher(
    val batch: java.util.ArrayList[String],
    val autoStart: Boolean,
    val showProgress: Boolean,
    val globalRounds: Int,
    val parallelism: Int,
    val seedName: String,
    val miniBatchSize: Int,
    val strategies: List[ExecutionStrategy[Any, Nothing]]
) extends Launcher {

  private val logger: Logger = LoggerFactory.getLogger(this.getClass.getName)
  private val errorQueue = new ConcurrentLinkedQueue[Throwable]()
  private val executor = Executors.newFixedThreadPool(parallelism)
  implicit private val executionContext: ExecutionContext = ExecutionContext.fromExecutor(executor)
  private var models: List[py.Dynamic] = List.empty
  private var targets: List[py.Dynamic] = List.empty

  override def launch(loader: Loader): Unit = {
    torch.manual_seed(42)
    val instances = loader.getVariables
    val prod = cartesianProduct(instances, batch)
    initializeNetwork()

    val experiences = List.fill(prod.size)(ExperienceBuffer[State](4000000))
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
      cleanPythonObjects(completedSimulations)
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

  private def neuralNetworkInjection(simulation: Simulation[Any, Nothing], iteration: Int): Unit = {
    val (model, _) = loadNetworks(iteration)
    val layer = new ModelLayer[Any, Nothing](simulation.getEnvironment, model)
    simulation.getEnvironment.addLayer(new SimpleMolecule(Molecules.model), layer)
  }

  private def initializeNetwork(): Unit = {
    val network = SimpleSequentialDQN(10, 256, ActionSpace.all.size)
    models = models :+ network
    targets = targets :+ network
  }

  private def saveNetworks(): Unit = {
    val path = "networks-snapshots/"
    Files.createDirectories(Paths.get(path))
    models.zipWithIndex.foreach { case (model, index) =>
      torch.save(model.state_dict(), s"${path}network-iteration-$index")
    }
  }

  private def collectExperience(simulations: List[Simulation[Any, Nothing]]): Seq[ExperienceBuffer[State]] = {
    simulations.map { simulation =>
      nodes(simulation)
        .map(_.getConcentration(new SimpleMolecule(Molecules.experience)))
        .map(_.asInstanceOf[ExperienceBuffer[State]])
        .map(_.getAll)
        .foldLeft(ExperienceBuffer[State](400000)) { (buffer, experience) =>
          buffer.addAll(experience)
          buffer
        }
    }
  }

  private val targetNetworkUpdateRate = 4 // TODO - refactor
  private val gamma = 0.9
  private val learningRate = 0.0005
  var ticks = 0
  private def improvePolicy(simulationsExperience: Seq[ExperienceBuffer[State]], iteration: Int): Unit = {
    // TODO - maybe this should be customizable with strategy or something similar
    println(s"Loading nn Iteration $iteration")
    val (actionNetwork, targetNetwork) = loadNetworks(iteration)
    val optimizer = torch.optim.Adam(actionNetwork.parameters(), learningRate)
    val losses: mutable.Map[Int, List[Double]] = mutable.Map.empty
    val allSize = simulationsExperience.map(_.getAll.size).sum
    val mergedBuffer = simulationsExperience.foldLeft(ExperienceBuffer[State](allSize)) { (buffer, experience) =>
      buffer.addAll(experience.getAll)
      buffer
    }
    val iterations = 300
    Range.inclusive(1, iterations).foreach { iter =>
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
      losses.get(0) match {
        case Some(l) => losses.update(0, l :+ totalLoss.item().as[Double])
        case None => losses.addOne(0 -> List(totalLoss.item().as[Double]))
      }

      optimizer.zero_grad()
      totalLoss.backward()
      torch.nn.utils.clip_grad_value_(actionNetwork.parameters(), 1.0)
      optimizer.step()
      if (ticks % targetNetworkUpdateRate == 0) {
        targetNetwork.load_state_dict(actionNetwork.state_dict())
      }

      logCsv(losses, iteration)
      models = models :+ actionNetwork
      targets = targets :+ targetNetwork
    }
  }

  private def logCsv(losses: mutable.Map[Int, List[Double]], iteration: Int): Unit = {
    val pd = py.module("pandas")
    val dataframe = pd.DataFrame(columns = List("time").toPythonProxy)
    losses.keys.foreach { k =>
      val values = losses(k).toPythonProxy
      dataframe.insert(1, s"Simulation-$k", values)
    }
    dataframe.to_csv(s"data/losses-iter-$iteration.csv")
  }

  private def loadNetworks(iteration: Int): (py.Dynamic, py.Dynamic) = {
    val actionNetwork = SimpleSequentialDQN(10, 256, ActionSpace.all.size)
    val targetNetwork = SimpleSequentialDQN(10, 256, ActionSpace.all.size)
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

  private def scheduleStrategies(
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
