package it.unibo.alchemist.boundary.launchers

import ai.djl.Model
import ai.djl.engine.Engine
import ai.djl.ndarray.types.{DataType, Shape}
import ai.djl.ndarray.{NDArray, NDList, NDManager}
import ai.djl.nn.Block
import ai.djl.training.{DefaultTrainingConfig, Trainer}
import ai.djl.training.loss.L2Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import ai.djl.translate.NoopTranslator
import it.unibo.alchemist.boundary.launchers.DeepQLearningLauncher.DQNFactory
import it.unibo.alchemist.core.Simulation
import it.unibo.alchemist.model.Node
import it.unibo.alchemist.model.layers.ModelLayer
import it.unibo.alchemist.model.learning._
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.experiment.{DJLContext, SimpleSequentialDQN}
import it.unibo.interop.PythonModules._
import me.shadaj.scalapy.py

import java.io.{DataOutputStream, FileOutputStream}
import java.nio.file.{Files, Paths}
import scala.jdk.CollectionConverters.{IteratorHasAsScala, MapHasAsScala}
import scala.util.Using

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
  private var models: List[Block] = List.empty
  private var targets: List[Block] = List.empty
  var handler = DJLContext.localManager()
  protected def neuralNetworkInjection(simulation: Simulation[Any, Nothing], iteration: Int): Unit = {
    val (model, _) = loadNetworks(iteration)
    val layer = new ModelLayer[Any, Nothing](simulation.getEnvironment, model)
    simulation.getEnvironment.addLayer(new SimpleMolecule(Molecules.model), layer)
  }

  protected def initializeNetwork(): Unit = {
    val network = networkFactory.create(DJLContext.globalManager)
    val target = networkFactory.create(DJLContext.globalManager)
    copyNetwork(network, target)
    models = models :+ network
    targets = targets :+ target
  }

  def copyNetwork(reference: Block, other: Block) = {
    val referenceMap = reference.getParameters.toMap().asScala
    val otherMap = other.getParameters.toMap().asScala
    referenceMap.foreach { case (key, value) =>
      otherMap(key).close()
      otherMap(key).setShape(null)
      otherMap(key).setArray(value.getArray.duplicate())
    }
  }
  protected def saveNetworks(): Unit = {
    val path = "networks-snapshots/"
    Files.createDirectories(Paths.get(path))
    models.zipWithIndex.foreach { case (model, index) =>
      val dataOutputStream = new DataOutputStream(new FileOutputStream(s"${path}network-iteration-$index"))
      model.saveParameters(dataOutputStream)
    }
  }

  private var ticks = 0
  protected def improvePolicy(simulationsExperience: Seq[ExperienceBuffer[State]], iteration: Int): Unit = {
    val (actionNetwork, targetNetwork) = loadNetworks(iteration)
    val allSize = simulationsExperience.map(_.getAll.size).sum
    val mergedBuffer = simulationsExperience.foldLeft(ExperienceBuffer[State](allSize)) { (buffer, experience) =>
      buffer.addAll(experience.getAll)
      buffer
    }
    val (modelAction, modelTarget) = (Model.newInstance("action"), Model.newInstance("target"))
    modelAction.setBlock(actionNetwork)
    modelTarget.setBlock(targetNetwork)
    val loss = new L2Loss()

    val optimizer = Optimizer.sgd().setLearningRateTracker(Tracker.fixed(0.003f)).build()
    val learningConfiguration = new DefaultTrainingConfig(loss)
      .optOptimizer(optimizer)
      .optDevices(handler.getEngine.getDevices(1))
    val actionPredictor = modelAction.newTrainer(learningConfiguration)
    val targetPredictor = modelTarget.newPredictor(new NoopTranslator())
    val iterations = mergedBuffer.getAll.size / learningInfo.miniBatchSize

    Range.inclusive(1, Math.min(iterations.toInt, learningInfo.iterations)).foreach { iter =>
      ticks += 1
      val (actualStateBatch, actionBatch, rewardBatch, nextStateBatch) =
        toBatches(mergedBuffer.sample(learningInfo.miniBatchSize))

      val pass = Using(Engine.getInstance().newGradientCollector()) { gc =>
        val networkPass = actionPredictor.forward(new NDList(actualStateBatch)).get(0)
        val stateActionValue = networkPass.gather(actionBatch.reshape(-1L, 1L), 0)
        val nextStateValues = targetPredictor.predict(new NDList(nextStateBatch)).get(0)
        val maxNextStateValues = nextStateValues.max(Array(1)).duplicate()
        val result = rewardBatch.add(maxNextStateValues.mul(learningInfo.gamma)).reshape(-1L, 1L)
        val lossDqn = loss.evaluate(new NDList(result), new NDList(stateActionValue))
        gc.backward(lossDqn)
      }
    }
    models = models :+ actionNetwork
    targets = targets :+ targetNetwork
  }

  protected def loadNetworks(iteration: Int): (Block, Block) = {
    val actionNetwork = networkFactory.create(DJLContext.globalManager)
    val targetNetwork = networkFactory.create(DJLContext.globalManager)
    copyNetwork(actionNetwork, models(iteration))
    copyNetwork(targetNetwork, targets(iteration))
    (actionNetwork, targetNetwork)
  }

  private def toBatches(experience: Seq[Experience[State]]): (NDArray, NDArray, NDArray, NDArray) = {
    val enconded = experience.map(_.encode)
    val actualStateBatch = handler.create(enconded.map(_._1.toArray).toArray)
    val actionBatch = handler.create(enconded.map(_._2).toArray)
    val rewardBatch = handler.create(enconded.map(_._3).toArray)
    val nextStateBatch = handler.create(enconded.map(_._4.toArray).toArray)
    (actualStateBatch, actionBatch, rewardBatch, nextStateBatch)
  }

  private def nodes(simulation: Simulation[Any, Nothing]): List[Node[Any]] =
    simulation.getEnvironment.getNodes.iterator().asScala.toList

  protected def cleanAfterRound(simulations: List[Simulation[Any, Nothing]]): Unit = {
    handler.close()
    handler = DJLContext.localManager()
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

  class DQNFactory(val input: Int, hidden: Int, output: Int) {
    def create(manager: NDManager): Block = SimpleSequentialDQN(input, hidden, output, manager)
  }
}
