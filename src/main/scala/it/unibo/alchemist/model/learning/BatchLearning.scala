package it.unibo.alchemist.model.learning

import it.unibo.alchemist.boundary.launchers.{DeepQLearningLauncher, SwarMDPBaseLauncher}

import scala.jdk.CollectionConverters.SeqHasAsJava
import it.unibo.alchemist.boundary.LoadAlchemist
import it.unibo.alchemist.boundary.launchers.DeepQLearningLauncher.LearningInfo

case class BatchLearning(
    strategies: List[ExecutionStrategy[Any, Nothing]],
    simulationConfiguration: String,
    batch: List[String],
    globalRounds: Int,
    parallelism: Int,
    miniBatchSize: Int,
    seedName: String,
    learningInfo: LearningInfo = DeepQLearningLauncher.LearningInfo(),
    networkFactory: DeepQLearningLauncher.DQNFactory
) {

  def startLearning(): Unit = {
    val loader = LoadAlchemist.from(simulationConfiguration)
    val launcher = new DeepQLearningLauncher(
      batch = new java.util.ArrayList(batch.asJava),
      globalRounds = globalRounds,
      parallelism = parallelism,
      seedName = seedName,
      strategies = strategies,
      globalSeed = 42,
      globalBufferSize = 4000000,
      learningInfo = learningInfo,
      networkFactory = networkFactory
    )
    launcher.launch(loader)
  }

}

object LearningLauncher {
  def apply(simulation: String, swarMDPLauncher: SwarMDPBaseLauncher): Unit = {
    val loader = LoadAlchemist.from(simulation)
    swarMDPLauncher.launch(loader)
  }
}